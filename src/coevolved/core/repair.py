"""Validation and repair utilities for LLM steps.

This module provides Instructor-style validation with automatic retry and repair
when parsing or validation fails. Failures are transformed into additional context
for subsequent LLM attempts.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

from pydantic import BaseModel, ValidationError

from coevolved.base.step import Step
from coevolved.core.types import LLMConfig, LLMProvider, LLMResponse, PromptPayload


FailureToInputFn = Callable[
    [Exception, Any, PromptPayload, LLMResponse, int],
    Union[str, Dict[str, Any], PromptPayload]
]
"""Type alias for failure-to-input transformation function.

Takes:
- failure: The exception that occurred (ValidationError, JSONDecodeError, etc.)
- state: Current state passed to the step
- prompt_payload: The prompt that was sent to the LLM
- response: The LLM response that failed validation
- attempt: Current attempt number (1-indexed)

Returns:
- New prompt input (str, dict, or PromptPayload) to use for next attempt
"""


@dataclass
class LLMRepairPolicy:
    """Policy for automatic validation repair and retry.
    
    When an LLM response fails validation or parsing, this policy controls
    how the failure is transformed into new input for retry attempts.
    
    Attributes:
        failure_to_input: Function that transforms failures into new prompts.
            If None, uses default_failure_to_input.
        max_attempts: Maximum number of attempts (default: 3).
        retryable_exceptions: Tuple of exception types to retry on.
            Default: (ValidationError, json.JSONDecodeError, ValueError)
        backoff_seconds: Initial delay between retries in seconds.
            If None, no delay is applied (default: None).
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0).
    
    Example:
        >>> policy = LLMRepairPolicy(
        ...     max_attempts=5,
        ...     backoff_seconds=1.0,
        ... )
        >>> # Custom repair function
        >>> def custom_repair(failure, state, prompt_payload, response, attempt):
        ...     return f"Previous error: {failure}. Try again."
        >>> custom_policy = LLMRepairPolicy(
        ...     failure_to_input=custom_repair,
        ...     max_attempts=3,
        ... )
    """
    failure_to_input: Optional[FailureToInputFn] = None
    max_attempts: int = 3
    retryable_exceptions: Tuple[type[BaseException], ...] = (
        ValidationError,
        json.JSONDecodeError,
        ValueError,
    )
    backoff_seconds: Optional[float] = None
    backoff_multiplier: float = 2.0


def default_failure_to_input(
    failure: Exception,
    state: Any,
    prompt_payload: PromptPayload,
    response: LLMResponse,
    attempt: int,
) -> Union[str, Dict[str, Any], PromptPayload]:
    """Default failure-to-input transformation.
    
    Appends the validation error to the prompt:
    - For text prompts: appends error message to text
    - For message-based prompts: adds a system message with error details
    
    Args:
        failure: The exception that occurred
        state: Current state
        prompt_payload: Original prompt payload
        response: The failed LLM response
        attempt: Current attempt number
    
    Returns:
        Updated prompt payload with error context
    """
    error_msg = f"Validation failed on attempt {attempt}: {str(failure)}"
    
    if isinstance(failure, ValidationError):
        # Include structured validation errors
        error_details = "\n".join(
            f"  - {err['loc']}: {err['msg']}"
            for err in failure.errors()
        )
        error_msg = f"{error_msg}\nErrors:\n{error_details}"
    
    # Add the failed response for context
    if response.text:
        error_msg += f"\n\nYour previous response:\n{response.text}"
    
    # Handle message-based prompts
    if prompt_payload.messages is not None:
        updated_messages = list(prompt_payload.messages)
        # Add the assistant's failed response
        if response.text:
            updated_messages.append({
                "role": "assistant",
                "content": response.text,
            })
        # Add system message with error
        updated_messages.append({
            "role": "system",
            "content": f"{error_msg}\n\nPlease correct the errors and try again.",
        })
        return PromptPayload(
            messages=updated_messages,
            prompt_id=prompt_payload.prompt_id,
            prompt_version=prompt_payload.prompt_version,
        )
    
    # Handle text-based prompts
    updated_text = prompt_payload.text or ""
    updated_text += f"\n\n{error_msg}\n\nPlease correct the errors and try again."
    
    return PromptPayload(
        text=updated_text,
        prompt_id=prompt_payload.prompt_id,
        prompt_version=prompt_payload.prompt_version,
    )

# Type alias for prompt builder (defined here to avoid circular import)
PromptBuilder = Callable[[Any], Union[str, "Prompt", PromptPayload, Dict[str, Any]]]  # noqa: F821


def validated_llm_step(
    *,
    prompt_builder: PromptBuilder,
    provider: LLMProvider,
    response_model: type[BaseModel],
    config: Optional[LLMConfig] = None,
    config_builder: Optional[Callable[[Any], LLMConfig]] = None,
    repair_policy: Optional[LLMRepairPolicy] = None,
    name: Optional[str] = None,
    annotations: Optional[Dict[str, Any]] = None,
    result_key: Optional[str] = "llm_response",
    log_prompt: bool = False,
) -> Step:
    """Create a Step that validates LLM responses with automatic repair/retry.
    
    Similar to llm_step, but adds Pydantic validation with automatic retry
    when validation fails. Failed attempts are transformed into additional
    context for subsequent attempts.
    
    Args:
        prompt_builder: Function that builds a prompt from state.
        provider: LLM provider implementation.
        response_model: Pydantic model for validating LLM responses.
        config: Static LLM configuration.
        config_builder: Function that builds LLM configuration from state.
            Exactly one of config or config_builder must be provided.
        repair_policy: Optional repair policy. Uses default if not provided.
        name: Optional step name. Defaults to "validated_llm_step".
        annotations: Optional metadata annotations.
        result_key: Key to attach result in state. If None, returns result directly.
        log_prompt: Whether to include full prompt text in events.
    
    Returns:
        Step that executes LLM call with validation and repair.
    
    Raises:
        ValueError: If validation fails after exhausting max_attempts.
    
    Example:
        >>> from pydantic import BaseModel, field_validator
        >>> 
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        ...     
        ...     @field_validator('age')
        ...     def validate_age(cls, v):
        ...         if v < 0:
        ...             raise ValueError('Age must be positive')
        ...         return v
        >>> 
        >>> step = validated_llm_step(
        ...     prompt_builder=lambda s: f"Extract user: {s['text']}",
        ...     provider=OpenAIProvider(...),
        ...     config=LLMConfig(model="gpt-4o-mini"),
        ...     response_model=User,
        ...     repair_policy=LLMRepairPolicy(max_attempts=3),
        ... )
    """
    policy = repair_policy or LLMRepairPolicy()
    failure_fn = policy.failure_to_input or default_failure_to_input
    
    # Parser that validates response and converts to Pydantic model
    def parse_and_validate(response: LLMResponse) -> BaseModel:
        """Parse and validate LLM response."""
        if not response.text:
            raise ValueError("LLM response contains no text to validate")
        
        # Try to parse as JSON first
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            # If not JSON, try to extract JSON from markdown code block
            text = response.text.strip()
            if "```json" in text:
                # Extract JSON from markdown code block
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_text = text[start:end].strip()
                data = json.loads(json_text)
            elif "```" in text:
                # Try generic code block
                start = text.find("```") + 3
                end = text.find("```", start)
                json_text = text[start:end].strip()
                data = json.loads(json_text)
            else:
                raise
        
        # Validate against Pydantic model
        return response_model.model_validate(data)
    
    def run(state: Any) -> Any:
        """Execute LLM step with validation and repair."""
        # Import here to avoid circular dependency
        from coevolved.core.llm import llm_step, _coerce_prompt_payload
        from coevolved.base.tracing import get_current_invocation, get_default_tracer
        
        current_state = state
        current_prompt_builder = prompt_builder
        last_error: Optional[Exception] = None
        delay = policy.backoff_seconds
        
        for attempt in range(1, policy.max_attempts + 1):
            try:
                # Create LLM step for this attempt
                attempt_annotations = {
                    **(annotations or {}),
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                }
                
                step = llm_step(
                    prompt_builder=current_prompt_builder,
                    provider=provider,
                    config=config,
                    config_builder=config_builder,
                    parser=parse_and_validate,
                    name=name or "validated_llm_step",
                    annotations=attempt_annotations,
                    result_key=result_key,
                    log_prompt=log_prompt,
                )
                
                # Execute the step
                return step(current_state)
                
            except policy.retryable_exceptions as exc:
                last_error = exc
                
                # Emit validation error event if we have invocation context
                invocation_ctx = get_current_invocation()
                if invocation_ctx:
                    tracer = get_default_tracer()
                    # Add validation error to annotations for debugging
                    from coevolved.core.events import LLMEvent
                    tracer.emit(
                        LLMEvent(
                            run_id=invocation_ctx.run_id,
                            step_id=invocation_ctx.step_id,
                            invocation_id=invocation_ctx.invocation_id,
                            group_hash=invocation_ctx.group_hash,
                            step_name=invocation_ctx.step_name,
                            event="validation_error",
                            timestamp=time.time(),
                            text=str(exc),
                            finish_reason="validation_failed",
                        )
                    )
                
                # If this was the last attempt, raise the error
                if attempt == policy.max_attempts:
                    raise ValueError(
                        f"Validation failed after {policy.max_attempts} attempts. "
                        f"Last error: {str(exc)}"
                    ) from exc
                
                # Get the original prompt payload for repair
                pb = current_prompt_builder(current_state)
                prompt_payload = _coerce_prompt_payload(pb, current_state)
                
                # Create a mock response for failure_fn (we need the failed text)
                # In a real scenario, we'd capture the actual response
                mock_response = LLMResponse(text="")
                
                # Transform failure into new input
                repaired_input = failure_fn(
                    exc,
                    current_state,
                    prompt_payload,
                    mock_response,
                    attempt,
                )
                
                # Update prompt builder to use repaired input
                current_prompt_builder = lambda s, inp=repaired_input: inp
                
                # Apply backoff if configured
                if delay is not None:
                    time.sleep(delay)
                    delay *= policy.backoff_multiplier
        
        # Should never reach here, but just in case
        if last_error:
            raise last_error
        return current_state
    
    return Step(
        run,
        name=name or "validated_llm_step",
        annotations={**(annotations or {}), "kind": "llm", "validated": True},
    )
