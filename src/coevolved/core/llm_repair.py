"""LLM validation and repair infrastructure.

This module provides the core abstraction for structured output validation
with automatic repair/retry when parsing or validation fails. Inspired by
the Instructor library's approach to reliable structured outputs.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from coevolved.base.step import Step
from coevolved.base.tracing import get_current_invocation, get_default_tracer
from coevolved.core.events import LLMEvent
from coevolved.core.types import LLMResponse, PromptPayload


class LLMValidationError(Exception):
    """Raised when LLM output validation fails after all retry attempts.
    
    Attributes:
        last_error: The final validation/parsing error that caused failure.
        attempts: Number of attempts made before giving up.
        last_response: The last LLM response received.
    """
    
    def __init__(
        self,
        message: str,
        *,
        last_error: Exception,
        attempts: int,
        last_response: Optional[LLMResponse] = None,
    ) -> None:
        super().__init__(message)
        self.last_error = last_error
        self.attempts = attempts
        self.last_response = last_response


@dataclass
class RepairContext:
    """Context passed to failure_to_input for repair decisions.
    
    Attributes:
        error: The exception that caused the failure (ValidationError, parsing error, etc.).
        state: Current state being processed.
        prompt_payload: The prompt payload that was sent to the LLM.
        response: The LLM response that failed validation.
        attempt: Current attempt number (1-indexed).
        max_attempts: Maximum attempts configured.
    """
    error: Exception
    state: Any
    prompt_payload: PromptPayload
    response: LLMResponse
    attempt: int
    max_attempts: int


@dataclass
class RepairResult:
    """Result from a repair function specifying how to modify the next attempt.
    
    At least one of the fields should be set. If multiple are set, they are
    applied in order: state updates, then prompt modifications.
    
    Attributes:
        state_updates: Dictionary of updates to merge into state.
        messages_append: Messages to append to the prompt messages list.
        prompt_text_append: Text to append to prompt text.
    """
    state_updates: Optional[Dict[str, Any]] = None
    messages_append: Optional[list] = None
    prompt_text_append: Optional[str] = None


FailureToInput = Callable[[RepairContext], RepairResult]
"""Type alias for repair functions.

A repair function receives context about the failure and returns
a RepairResult specifying how to modify the next attempt.
"""


def default_validation_repair(ctx: RepairContext) -> RepairResult:
    """Default repair function that appends validation errors to messages.
    
    For ValidationError, formats the error details in a structured way.
    For other errors, includes the error message.
    
    Args:
        ctx: Repair context with failure details.
    
    Returns:
        RepairResult with error context appended as a user message.
    """
    if isinstance(ctx.error, ValidationError):
        error_details = []
        for err in ctx.error.errors():
            loc = ".".join(str(x) for x in err["loc"]) if err["loc"] else "root"
            error_details.append(f"- {loc}: {err['msg']}")
        error_text = "\n".join(error_details)
        repair_message = (
            f"Your previous response failed validation. Please fix the following errors "
            f"and try again:\n{error_text}\n\n"
            f"Previous response that failed:\n{ctx.response.text or '[no text]'}"
        )
    else:
        repair_message = (
            f"Your previous response could not be parsed: {ctx.error}\n\n"
            f"Please provide a valid response. Previous response:\n"
            f"{ctx.response.text or '[no text]'}"
        )
    
    return RepairResult(
        messages_append=[{"role": "user", "content": repair_message}]
    )


DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ValidationError,
    ValueError,
    TypeError,
    KeyError,
)
"""Default exception types that trigger retry with repair."""


@dataclass
class LLMRepairPolicy:
    """Policy for LLM validation and repair behavior.
    
    Configures how validation failures are handled, including retry limits,
    repair strategies, and backoff behavior.
    
    Attributes:
        max_attempts: Maximum number of attempts (including the initial attempt).
        failure_to_input: Function that transforms failure into repair instructions.
            Defaults to default_validation_repair which appends errors as messages.
        retryable_exceptions: Tuple of exception types that should trigger retry.
        backoff_seconds: Initial delay between retries (0 for no delay).
        backoff_multiplier: Multiplier for exponential backoff.
        include_response_in_repair: Whether to include the failed response text
            in repair context (may be disabled for privacy).
    
    Example:
        >>> policy = LLMRepairPolicy(
        ...     max_attempts=3,
        ...     failure_to_input=my_custom_repair,
        ...     backoff_seconds=0.5,
        ... )
    """
    max_attempts: int = 3
    failure_to_input: FailureToInput = field(default=default_validation_repair)
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS
    backoff_seconds: float = 0.0
    backoff_multiplier: float = 2.0
    include_response_in_repair: bool = True

    def is_retryable(self, exc: Exception) -> bool:
        """Check if an exception should trigger a retry.
        
        Args:
            exc: The exception to check.
        
        Returns:
            True if the exception type is in retryable_exceptions.
        """
        return isinstance(exc, self.retryable_exceptions)


@dataclass
class ValidationAttemptEvent:
    """Event data for a validation attempt (used in annotations).
    
    Attributes:
        attempt: Attempt number (1-indexed).
        max_attempts: Maximum attempts configured.
        success: Whether this attempt succeeded.
        error: Error message if failed.
        elapsed_ms: Time taken for this attempt.
    """
    attempt: int
    max_attempts: int
    success: bool
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for event annotations."""
        return {
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "success": self.success,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


def apply_repair_result(
    state: Any,
    prompt_payload: PromptPayload,
    repair: RepairResult,
) -> Tuple[Any, PromptPayload]:
    """Apply a RepairResult to state and prompt payload.
    
    Args:
        state: Current state.
        prompt_payload: Current prompt payload.
        repair: Repair result to apply.
    
    Returns:
        Tuple of (updated_state, updated_prompt_payload).
    """
    new_state = state
    new_payload = prompt_payload
    
    if repair.state_updates:
        if isinstance(state, dict):
            new_state = {**state, **repair.state_updates}
        elif isinstance(state, BaseModel):
            new_state = state.model_copy(update=repair.state_updates)
        else:
            for k, v in repair.state_updates.items():
                setattr(new_state, k, v)
    
    if repair.messages_append and new_payload.messages is not None:
        new_messages = list(new_payload.messages) + repair.messages_append
        new_payload = new_payload.model_copy(update={"messages": new_messages})
    elif repair.messages_append and new_payload.messages is None:
        if new_payload.text:
            initial_messages = [{"role": "user", "content": new_payload.text}]
            new_messages = initial_messages + repair.messages_append
            new_payload = new_payload.model_copy(
                update={"messages": new_messages, "text": None}
            )
    
    if repair.prompt_text_append and new_payload.text is not None:
        new_text = new_payload.text + "\n\n" + repair.prompt_text_append
        new_payload = new_payload.model_copy(update={"text": new_text})
    
    return new_state, new_payload


def validated_llm_call(
    *,
    llm_fn: Callable[[Any, PromptPayload], LLMResponse],
    prompt_payload: PromptPayload,
    state: Any,
    validator: Callable[[LLMResponse], Any],
    policy: LLMRepairPolicy,
    step_name: str = "validated_llm",
) -> Any:
    """Execute an LLM call with validation and automatic repair/retry.
    
    This is the core function that implements the validation loop. It:
    1. Calls the LLM
    2. Validates the response
    3. On validation failure, applies repair and retries
    4. Returns the validated output or raises LLMValidationError
    
    Args:
        llm_fn: Function that executes the LLM call. Signature: (state, payload) -> LLMResponse
        prompt_payload: Initial prompt payload.
        state: Current state.
        validator: Function that validates LLMResponse and returns parsed output.
            Should raise an exception (typically ValidationError) on failure.
        policy: Repair policy configuration.
        step_name: Name for tracing purposes.
    
    Returns:
        Validated and parsed output from the validator function.
    
    Raises:
        LLMValidationError: If validation fails after all attempts.
    """
    tracer = get_default_tracer()
    invocation_ctx = get_current_invocation()
    
    current_state = state
    current_payload = prompt_payload
    last_error: Optional[Exception] = None
    last_response: Optional[LLMResponse] = None
    delay = policy.backoff_seconds
    
    for attempt in range(1, policy.max_attempts + 1):
        attempt_start = time.perf_counter()
        
        try:
            response = llm_fn(current_state, current_payload)
            last_response = response
            
            result = validator(response)
            
            if invocation_ctx:
                attempt_event = ValidationAttemptEvent(
                    attempt=attempt,
                    max_attempts=policy.max_attempts,
                    success=True,
                    elapsed_ms=(time.perf_counter() - attempt_start) * 1000,
                )
                tracer.emit(
                    LLMEvent(
                        run_id=invocation_ctx.run_id,
                        step_id=invocation_ctx.step_id,
                        invocation_id=invocation_ctx.invocation_id,
                        group_hash=invocation_ctx.group_hash,
                        step_name=step_name,
                        event="validation_success",
                        timestamp=time.time(),
                        text=response.text,
                        finish_reason=response.finish_reason,
                        model=response.model,
                        usage=response.usage,
                    )
                )
            
            return result
            
        except policy.retryable_exceptions as exc:
            last_error = exc
            elapsed_ms = (time.perf_counter() - attempt_start) * 1000
            
            if invocation_ctx:
                attempt_event = ValidationAttemptEvent(
                    attempt=attempt,
                    max_attempts=policy.max_attempts,
                    success=False,
                    error=str(exc),
                    elapsed_ms=elapsed_ms,
                )
                tracer.emit(
                    LLMEvent(
                        run_id=invocation_ctx.run_id,
                        step_id=invocation_ctx.step_id,
                        invocation_id=invocation_ctx.invocation_id,
                        group_hash=invocation_ctx.group_hash,
                        step_name=step_name,
                        event="validation_failure",
                        timestamp=time.time(),
                        text=last_response.text if last_response else None,
                        finish_reason=last_response.finish_reason if last_response else None,
                        model=last_response.model if last_response else None,
                    )
                )
            
            if attempt >= policy.max_attempts:
                break
            
            repair_ctx = RepairContext(
                error=exc,
                state=current_state,
                prompt_payload=current_payload,
                response=last_response or LLMResponse(),
                attempt=attempt,
                max_attempts=policy.max_attempts,
            )
            
            repair_result = policy.failure_to_input(repair_ctx)
            current_state, current_payload = apply_repair_result(
                current_state, current_payload, repair_result
            )
            
            if delay > 0:
                time.sleep(delay)
                delay *= policy.backoff_multiplier
    
    raise LLMValidationError(
        f"Validation failed after {policy.max_attempts} attempts: {last_error}",
        last_error=last_error or ValueError("Unknown error"),
        attempts=policy.max_attempts,
        last_response=last_response,
    )
