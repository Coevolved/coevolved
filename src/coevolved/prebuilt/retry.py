"""High-level retry helpers for LLM and agent steps.

This module provides convenient wrappers for adding validation/repair and
custom retry logic to LLM and agent steps.
"""

import time
from typing import Any, Callable, Optional

from pydantic import BaseModel

from coevolved.base.step import Step
from coevolved.core.repair import LLMRepairPolicy, validated_llm_step
from coevolved.core.types import LLMConfig, LLMProvider


def llm_retry(
    *,
    prompt_builder: Callable[[Any], Any],
    provider: LLMProvider,
    response_model: type[BaseModel],
    config: Optional[LLMConfig] = None,
    config_builder: Optional[Callable[[Any], LLMConfig]] = None,
    repair_policy: Optional[LLMRepairPolicy] = None,
    max_attempts: int = 3,
    name: Optional[str] = None,
    result_key: Optional[str] = "llm_response",
) -> Step:
    """Create an LLM step with automatic validation and repair.
    
    Convenience wrapper around validated_llm_step that provides a simpler
    API for common use cases.
    
    Args:
        prompt_builder: Function that builds a prompt from state.
        provider: LLM provider implementation.
        response_model: Pydantic model for validating responses.
        config: Static LLM configuration.
        config_builder: Function that builds LLM configuration from state.
        repair_policy: Optional custom repair policy. If not provided,
            creates a default policy with max_attempts.
        max_attempts: Maximum retry attempts (default: 3).
        name: Optional step name.
        result_key: Key to attach result in state.
    
    Returns:
        Step with validation and repair logic.
    
    Example:
        >>> from pydantic import BaseModel
        >>> 
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> 
        >>> step = llm_retry(
        ...     prompt_builder=lambda s: f"Extract: {s['text']}",
        ...     provider=OpenAIProvider(...),
        ...     config=LLMConfig(model="gpt-4o-mini"),
        ...     response_model=User,
        ...     max_attempts=5,
        ... )
    """
    policy = repair_policy or LLMRepairPolicy(max_attempts=max_attempts)
    
    return validated_llm_step(
        prompt_builder=prompt_builder,
        provider=provider,
        response_model=response_model,
        config=config,
        config_builder=config_builder,
        repair_policy=policy,
        name=name or "llm_retry",
        result_key=result_key,
    )


def agent_retry(
    step: Step,
    *,
    should_retry: Callable[[Exception, Any, int], bool],
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    backoff_multiplier: float = 2.0,
) -> Step:
    """Wrap an agent step with custom retry logic.
    
    Provides fine-grained control over retry behavior using a custom
    predicate function. Useful for agent workflows where you want to
    retry based on specific error conditions or state.
    
    Args:
        step: Agent step or callable to wrap.
        should_retry: Predicate function that determines if retry should occur.
            Takes (exception, state, attempt_number) and returns bool.
        max_attempts: Maximum retry attempts (default: 3).
        backoff_seconds: Initial delay between retries (default: 0.5).
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0).
    
    Returns:
        Wrapped step with retry logic.
    
    Example:
        >>> def should_retry_fn(exc, state, attempt):
        ...     # Only retry on specific errors
        ...     if isinstance(exc, ValueError):
        ...         return attempt < 3
        ...     return False
        >>> 
        >>> retry_step = agent_retry(
        ...     my_agent_step,
        ...     should_retry=should_retry_fn,
        ...     max_attempts=5,
        ... )
    """
    def runner(state: Any) -> Any:
        delay = backoff_seconds
        last_error: Optional[Exception] = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                return step(state)
            except Exception as exc:
                last_error = exc
                
                # Check if we should retry
                if not should_retry(exc, state, attempt):
                    raise
                
                # If this was the last attempt, raise
                if attempt == max_attempts:
                    raise
                
                # Apply backoff
                time.sleep(delay)
                delay *= backoff_multiplier
        
        # Should never reach here, but just in case
        if last_error:
            raise last_error
        return state
    
    return Step(
        runner,
        name=f"{step.name}_agent_retry",
        input_schema=step.input_schema,
        output_schema=step.output_schema,
        annotations={**step.annotations, "agent_retry_wrapper": True},
    )
