"""LLM and agent retry utilities.

This module provides prebuilt helpers for wrapping LLM steps and agents
with validation, repair, and retry behavior.
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from coevolved.base.step import Step
from coevolved.base.tracing import get_default_tracer
from coevolved.core.llm_repair import (
    DEFAULT_RETRYABLE_EXCEPTIONS,
    LLMRepairPolicy,
    LLMValidationError,
    RepairContext,
    RepairResult,
    apply_repair_result,
    default_validation_repair,
    validated_llm_call,
)
from coevolved.core.types import LLMResponse, PromptPayload


def llm_retry(
    step: Step,
    *,
    output_schema: Optional[Type[BaseModel]] = None,
    validator: Optional[Callable[[LLMResponse], Any]] = None,
    repair_policy: Optional[LLMRepairPolicy] = None,
    max_attempts: int = 3,
    failure_to_input: Optional[Callable[[RepairContext], RepairResult]] = None,
    result_key: Optional[str] = "validated_output",
) -> Step:
    """Wrap an LLM step with validation and automatic repair/retry.
    
    This creates a new Step that executes the original LLM step, validates
    the output, and retries with repair context if validation fails.
    
    The validation can be specified either via:
    - output_schema: A Pydantic model to validate against (parses response.text as JSON)
    - validator: A custom function that validates LLMResponse and returns parsed output
    
    If neither is provided, the step will just retry on any exception from the
    original step's parser.
    
    Args:
        step: The LLM step to wrap. Must be created with llm_step().
        output_schema: Optional Pydantic model for output validation.
            If provided, response.text is parsed as JSON and validated.
        validator: Optional custom validator function.
            Signature: (LLMResponse) -> Any. Should raise on validation failure.
        repair_policy: Full repair policy configuration. If provided, overrides
            max_attempts and failure_to_input.
        max_attempts: Maximum number of attempts (default: 3).
        failure_to_input: Custom repair function. Defaults to default_validation_repair.
        result_key: Key to attach validated result in state. If None, returns
            result directly. Defaults to "validated_output".
    
    Returns:
        Step that executes the LLM call with validation and retry.
    
    Raises:
        LLMValidationError: If validation fails after all attempts.
        ValueError: If step is not an LLM step.
    
    Example:
        >>> from pydantic import BaseModel
        >>> 
        >>> class ExtractedData(BaseModel):
        ...     name: str
        ...     age: int
        >>> 
        >>> # Create base LLM step
        >>> llm = llm_step(
        ...     prompt_builder=lambda s: f"Extract: {s['text']}",
        ...     provider=provider,
        ...     config=config,
        ... )
        >>> 
        >>> # Wrap with validation and retry
        >>> validated_llm = llm_retry(
        ...     llm,
        ...     output_schema=ExtractedData,
        ...     max_attempts=3,
        ... )
        >>> 
        >>> result = validated_llm({"text": "John is 25 years old"})
        >>> # result["validated_output"] is ExtractedData(name="John", age=25)
    
    Example with custom validator:
        >>> def validate_and_parse(response: LLMResponse) -> dict:
        ...     data = json.loads(response.text)
        ...     if "error" in data:
        ...         raise ValueError(f"LLM returned error: {data['error']}")
        ...     return data
        >>> 
        >>> validated_llm = llm_retry(
        ...     llm,
        ...     validator=validate_and_parse,
        ...     max_attempts=2,
        ... )
    
    Example with custom repair:
        >>> def my_repair(ctx: RepairContext) -> RepairResult:
        ...     # Add specific guidance based on error type
        ...     if "age" in str(ctx.error):
        ...         hint = "Age must be a positive integer."
        ...     else:
        ...         hint = "Please fix the errors and try again."
        ...     return RepairResult(
        ...         messages_append=[{"role": "user", "content": hint}]
        ...     )
        >>> 
        >>> validated_llm = llm_retry(
        ...     llm,
        ...     output_schema=ExtractedData,
        ...     failure_to_input=my_repair,
        ... )
    """
    if step.annotations.get("kind") != "llm":
        raise ValueError("llm_retry requires an LLM step created with llm_step().")
    
    policy = repair_policy or LLMRepairPolicy(
        max_attempts=max_attempts,
        failure_to_input=failure_to_input or default_validation_repair,
    )
    
    actual_validator = _build_validator(output_schema, validator)
    
    def run(state: Any) -> Any:
        last_error: Optional[Exception] = None
        last_response: Optional[LLMResponse] = None
        current_state = state
        delay = policy.backoff_seconds
        
        for attempt in range(1, policy.max_attempts + 1):
            try:
                result_state = step(current_state)
                
                llm_response_key = step.annotations.get("result_key", "llm_response")
                if isinstance(result_state, dict):
                    response = result_state.get(llm_response_key)
                elif isinstance(result_state, BaseModel):
                    response = getattr(result_state, llm_response_key, None)
                else:
                    response = getattr(result_state, llm_response_key, None)
                
                if response is None:
                    response = result_state
                
                if not isinstance(response, LLMResponse):
                    if isinstance(response, dict):
                        response = LLMResponse.model_validate(response)
                    else:
                        response = LLMResponse(text=str(response))
                
                last_response = response
                validated = actual_validator(response)
                
                if result_key is None:
                    return validated
                if isinstance(result_state, dict):
                    return {**result_state, result_key: validated}
                if isinstance(result_state, BaseModel):
                    return result_state.model_copy(update={result_key: validated})
                return validated
                
            except policy.retryable_exceptions as exc:
                last_error = exc
                
                if attempt >= policy.max_attempts:
                    break
                
                repair_ctx = RepairContext(
                    error=exc,
                    state=current_state,
                    prompt_payload=PromptPayload(),
                    response=last_response or LLMResponse(),
                    attempt=attempt,
                    max_attempts=policy.max_attempts,
                )
                
                repair_result = policy.failure_to_input(repair_ctx)
                
                if repair_result.state_updates:
                    if isinstance(current_state, dict):
                        current_state = {**current_state, **repair_result.state_updates}
                    elif isinstance(current_state, BaseModel):
                        current_state = current_state.model_copy(
                            update=repair_result.state_updates
                        )
                
                if repair_result.messages_append:
                    messages_key = "messages"
                    if isinstance(current_state, dict):
                        existing = current_state.get(messages_key, [])
                        current_state = {
                            **current_state,
                            messages_key: list(existing) + repair_result.messages_append,
                        }
                    elif isinstance(current_state, BaseModel):
                        existing = getattr(current_state, messages_key, []) or []
                        current_state = current_state.model_copy(
                            update={messages_key: list(existing) + repair_result.messages_append}
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
    
    return Step(
        run,
        name=f"{step.name}_retry",
        input_schema=step.input_schema,
        output_schema=step.output_schema,
        annotations={
            **step.annotations,
            "retry_wrapper": True,
            "max_attempts": policy.max_attempts,
        },
    )


def _build_validator(
    output_schema: Optional[Type[BaseModel]],
    validator: Optional[Callable[[LLMResponse], Any]],
) -> Callable[[LLMResponse], Any]:
    """Build the validator function from schema or custom validator."""
    if validator is not None:
        return validator
    
    if output_schema is not None:
        import json
        
        def schema_validator(response: LLMResponse) -> BaseModel:
            if response.text is None:
                raise ValueError("LLM response has no text to parse")
            
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}") from e
            
            return output_schema.model_validate(data)
        
        return schema_validator
    
    def passthrough(response: LLMResponse) -> LLMResponse:
        return response
    
    return passthrough


ShouldRetry = Callable[[Exception, Any, int], bool]
"""Type alias for retry predicate functions.

Signature: (exception, state, attempt) -> should_retry
"""


def default_should_retry(exc: Exception, state: Any, attempt: int) -> bool:
    """Default retry predicate that retries on common retryable exceptions."""
    return isinstance(exc, DEFAULT_RETRYABLE_EXCEPTIONS)


def agent_retry(
    step: Step,
    *,
    max_attempts: int = 3,
    should_retry: Optional[ShouldRetry] = None,
    backoff_seconds: float = 0.0,
    backoff_multiplier: float = 2.0,
    on_retry: Optional[Callable[[Exception, Any, int], Any]] = None,
) -> Step:
    """Wrap a step or agent with bounded retry logic.
    
    This is a higher-level retry wrapper suitable for agent workflows.
    Unlike llm_retry which focuses on LLM output validation, agent_retry
    provides general-purpose retry with customizable predicates.
    
    Args:
        step: The step or agent to wrap.
        max_attempts: Maximum number of attempts (default: 3).
        should_retry: Predicate function to determine if an exception should
            trigger a retry. Signature: (exc, state, attempt) -> bool.
            Defaults to retrying on ValidationError and common parsing errors.
        backoff_seconds: Initial delay between retries (default: 0).
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0).
        on_retry: Optional callback invoked before each retry.
            Signature: (exc, state, attempt) -> new_state.
            Can be used to modify state between retries. If it returns a value,
            that becomes the new state for the retry.
    
    Returns:
        Step that executes with retry logic.
    
    Raises:
        Exception: The last exception if all attempts fail.
    
    Example:
        >>> agent = react_agent(planner=planner, tools=tools)
        >>> 
        >>> # Retry the agent up to 3 times on failure
        >>> reliable_agent = agent_retry(agent, max_attempts=3)
        >>> result = reliable_agent(initial_state)
    
    Example with custom retry predicate:
        >>> def retry_on_timeout(exc, state, attempt):
        ...     return isinstance(exc, TimeoutError) and attempt < 5
        >>> 
        >>> reliable_agent = agent_retry(
        ...     agent,
        ...     should_retry=retry_on_timeout,
        ...     backoff_seconds=1.0,
        ... )
    
    Example with state modification on retry:
        >>> def reset_on_retry(exc, state, attempt):
        ...     # Clear tool results and try again
        ...     return {**state, "tool_result": None, "retry_count": attempt}
        >>> 
        >>> reliable_agent = agent_retry(
        ...     agent,
        ...     on_retry=reset_on_retry,
        ... )
    """
    retry_predicate = should_retry or default_should_retry
    
    def run(state: Any) -> Any:
        current_state = state
        delay = backoff_seconds
        last_error: Optional[Exception] = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                return step(current_state)
            except Exception as exc:
                last_error = exc
                
                if attempt >= max_attempts:
                    raise
                
                if not retry_predicate(exc, current_state, attempt):
                    raise
                
                if on_retry is not None:
                    modified = on_retry(exc, current_state, attempt)
                    if modified is not None:
                        current_state = modified
                
                if delay > 0:
                    time.sleep(delay)
                    delay *= backoff_multiplier
        
        if last_error:
            raise last_error
        return current_state
    
    return Step(
        run,
        name=f"{step.name}_retry",
        input_schema=step.input_schema,
        output_schema=step.output_schema,
        annotations={
            **step.annotations,
            "agent_retry_wrapper": True,
            "max_attempts": max_attempts,
        },
    )
