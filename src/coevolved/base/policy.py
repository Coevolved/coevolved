"""Execution policies for budget and timeout enforcement.

This module provides UsagePolicy and UsageTracker for enforcing
resource limits (time, steps, tokens, cost) during workflow execution.
"""

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional


class BudgetExceededError(Exception):
    """Raised when an execution budget limit is exceeded.
    
    Attributes:
        limit_type: The type of limit that was exceeded.
        limit: The configured limit value.
        current: The current value that exceeded the limit.
    """
    
    def __init__(self, limit_type: str, limit: Any, current: Any) -> None:
        self.limit_type = limit_type
        self.limit = limit
        self.current = current
        super().__init__(f"{limit_type} exceeded: {current} >= {limit}")


class TimeoutExceededError(BudgetExceededError):
    """Raised when execution timeout is exceeded."""
    
    def __init__(self, timeout_seconds: float, elapsed_seconds: float) -> None:
        super().__init__("timeout", timeout_seconds, elapsed_seconds)


@dataclass
class UsagePolicy:
    """Constraints for workflow execution.
    
    Defines limits on steps, LLM calls, tool calls, time, tokens, and cost.
    Used with UsageTracker to enforce these limits during execution.
    
    Attributes:
        max_steps: Maximum total steps to execute.
        max_llm_calls: Maximum LLM invocations.
        max_tool_calls: Maximum tool invocations.
        timeout_seconds: Maximum wall-clock time for execution.
        max_tokens: Maximum total tokens used (requires usage tracking).
        max_cost_dollars: Maximum cost in dollars (requires cost tracking).
        on_limit: Behavior when limit is reached - "raise" or "return".
        on_step: Optional callback called after each step.
        on_warning: Optional callback for budget warnings (e.g., 80% used).
        warning_threshold: Fraction of budget at which to trigger warnings.
    
    Example:
        >>> policy = UsagePolicy(
        ...     max_steps=50,
        ...     max_llm_calls=20,
        ...     timeout_seconds=120.0,
        ...     max_cost_dollars=1.0,
        ... )
    """
    # Iteration limits
    max_steps: int = 100
    max_llm_calls: int = 50
    max_tool_calls: int = 200
    
    # Time limits
    timeout_seconds: float = 300.0
    
    # Cost limits (requires usage tracking)
    max_tokens: Optional[int] = None
    max_cost_dollars: Optional[float] = None
    
    # Behavior on limit
    on_limit: Literal["raise", "return"] = "raise"
    
    # Callbacks
    on_step: Optional[Callable[[str, int], None]] = None
    on_warning: Optional[Callable[[str, float], None]] = None
    warning_threshold: float = 0.8
    
    def with_overrides(self, **kwargs: Any) -> "UsagePolicy":
        """Create a copy with specified overrides.
        
        Args:
            **kwargs: Fields to override.
        
        Returns:
            New UsagePolicy with overrides applied.
        """
        return UsagePolicy(
            max_steps=kwargs.get("max_steps", self.max_steps),
            max_llm_calls=kwargs.get("max_llm_calls", self.max_llm_calls),
            max_tool_calls=kwargs.get("max_tool_calls", self.max_tool_calls),
            timeout_seconds=kwargs.get("timeout_seconds", self.timeout_seconds),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            max_cost_dollars=kwargs.get("max_cost_dollars", self.max_cost_dollars),
            on_limit=kwargs.get("on_limit", self.on_limit),
            on_step=kwargs.get("on_step", self.on_step),
            on_warning=kwargs.get("on_warning", self.on_warning),
            warning_threshold=kwargs.get("warning_threshold", self.warning_threshold),
        )


@dataclass
class UsageTracker:
    """Tracks execution usage against policies.
    
    Accumulates usage metrics and checks them against policy limits.
    Should be created at the start of workflow execution and updated
    as steps execute.
    
    Attributes:
        policy: The UsagePolicy to enforce.
        step_count: Number of steps executed.
        llm_calls: Number of LLM calls made.
        tool_calls: Number of tool calls made.
        tokens_used: Total tokens consumed.
        cost_dollars: Total cost in dollars.
        start_time: Unix timestamp when execution started.
    
    Example:
        >>> tracker = UsageTracker(policy)
        >>> tracker.check("llm")  # Check if LLM call is allowed
        >>> tracker.record_step("llm")  # Record the call
        >>> tracker.record_usage(tokens=500, cost=0.01)
    """
    policy: UsagePolicy
    step_count: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    cost_dollars: float = 0.0
    start_time: float = field(default_factory=time.time)
    _warnings_issued: set = field(default_factory=set)
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since execution started."""
        return time.time() - self.start_time
    
    @property
    def remaining_seconds(self) -> float:
        """Get remaining time before timeout."""
        return max(0, self.policy.timeout_seconds - self.elapsed_seconds)
    
    def check(self, operation: str = "step") -> bool:
        """Check if budget allows an operation.
        
        Args:
            operation: Type of operation - "step", "llm", or "tool".
        
        Returns:
            True if operation is allowed.
        
        Raises:
            BudgetExceededError: If policy.on_limit is "raise" and limit exceeded.
            TimeoutExceededError: If timeout exceeded.
        """
        # Check timeout
        elapsed = self.elapsed_seconds
        if elapsed > self.policy.timeout_seconds:
            return self._handle_limit(
                TimeoutExceededError(self.policy.timeout_seconds, elapsed)
            )
        
        # Check step count
        if self.step_count >= self.policy.max_steps:
            return self._handle_limit(
                BudgetExceededError("max_steps", self.policy.max_steps, self.step_count)
            )
        
        # Check operation-specific limits
        if operation == "llm" and self.llm_calls >= self.policy.max_llm_calls:
            return self._handle_limit(
                BudgetExceededError("max_llm_calls", self.policy.max_llm_calls, self.llm_calls)
            )
        
        if operation == "tool" and self.tool_calls >= self.policy.max_tool_calls:
            return self._handle_limit(
                BudgetExceededError("max_tool_calls", self.policy.max_tool_calls, self.tool_calls)
            )
        
        # Check token limit
        if self.policy.max_tokens and self.tokens_used >= self.policy.max_tokens:
            return self._handle_limit(
                BudgetExceededError("max_tokens", self.policy.max_tokens, self.tokens_used)
            )
        
        # Check cost limit
        if self.policy.max_cost_dollars and self.cost_dollars >= self.policy.max_cost_dollars:
            return self._handle_limit(
                BudgetExceededError("max_cost", self.policy.max_cost_dollars, self.cost_dollars)
            )
        
        # Check for warnings
        self._check_warnings()
        
        return True
    
    def record_step(self, step_kind: Optional[str] = None) -> None:
        """Record a step execution.
        
        Args:
            step_kind: Type of step - "llm", "tool", or None for generic.
        """
        self.step_count += 1
        if step_kind == "llm":
            self.llm_calls += 1
        elif step_kind == "tool":
            self.tool_calls += 1
        
        if self.policy.on_step:
            self.policy.on_step(step_kind or "step", self.step_count)
    
    def record_usage(self, tokens: int = 0, cost: float = 0.0) -> None:
        """Record token and cost usage.
        
        Args:
            tokens: Number of tokens used.
            cost: Cost in dollars.
        """
        self.tokens_used += tokens
        self.cost_dollars += cost
    
    def _handle_limit(self, error: BudgetExceededError) -> bool:
        """Handle a limit being exceeded.
        
        Args:
            error: The error describing the limit.
        
        Returns:
            False if on_limit is "return".
        
        Raises:
            BudgetExceededError: If on_limit is "raise".
        """
        if self.policy.on_limit == "raise":
            raise error
        return False
    
    def _check_warnings(self) -> None:
        """Check if any budget warnings should be issued."""
        if not self.policy.on_warning:
            return
        
        threshold = self.policy.warning_threshold
        
        # Check step warning
        if self.step_count >= self.policy.max_steps * threshold:
            if "steps" not in self._warnings_issued:
                self._warnings_issued.add("steps")
                ratio = self.step_count / self.policy.max_steps
                self.policy.on_warning("steps", ratio)
        
        # Check LLM call warning
        if self.llm_calls >= self.policy.max_llm_calls * threshold:
            if "llm_calls" not in self._warnings_issued:
                self._warnings_issued.add("llm_calls")
                ratio = self.llm_calls / self.policy.max_llm_calls
                self.policy.on_warning("llm_calls", ratio)
        
        # Check timeout warning
        elapsed = self.elapsed_seconds
        if elapsed >= self.policy.timeout_seconds * threshold:
            if "timeout" not in self._warnings_issued:
                self._warnings_issued.add("timeout")
                ratio = elapsed / self.policy.timeout_seconds
                self.policy.on_warning("timeout", ratio)
    
    def summary(self) -> dict:
        """Get a summary of execution metrics.
        
        Returns:
            Dictionary with all metrics and their limits.
        """
        return {
            "step_count": self.step_count,
            "max_steps": self.policy.max_steps,
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.policy.max_llm_calls,
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.policy.max_tool_calls,
            "tokens_used": self.tokens_used,
            "max_tokens": self.policy.max_tokens,
            "cost_dollars": self.cost_dollars,
            "max_cost_dollars": self.policy.max_cost_dollars,
            "elapsed_seconds": self.elapsed_seconds,
            "timeout_seconds": self.policy.timeout_seconds,
        }


# Context variable for current usage tracker
_current_usage_tracker: ContextVar[Optional[UsageTracker]] = ContextVar(
    "current_usage_tracker", default=None
)


def get_usage_tracker() -> Optional[UsageTracker]:
    """Get the current usage tracker.
    
    Returns:
        The current UsageTracker, or None if not in a policy-controlled execution.
    """
    return _current_usage_tracker.get()


def set_usage_tracker(tracker: UsageTracker) -> None:
    """Set the current usage tracker.
    
    Args:
        tracker: The UsageTracker to set.
    """
    _current_usage_tracker.set(tracker)


def clear_usage_tracker() -> None:
    """Clear the current usage tracker."""
    _current_usage_tracker.set(None)


def check_budget(operation: str = "step") -> bool:
    """Check if current usage tracker allows an operation.
    
    Convenience function that checks the current context if one exists.
    If no context is set, returns True (no limits).
    
    Args:
        operation: Type of operation - "step", "llm", or "tool".
    
    Returns:
        True if operation is allowed (or no context set).
    
    Raises:
        BudgetExceededError: If limit exceeded and policy.on_limit is "raise".
    """
    tracker = get_usage_tracker()
    if tracker is None:
        return True
    return tracker.check(operation)


def record_step(step_kind: Optional[str] = None) -> None:
    """Record a step in the current usage tracker.
    
    Convenience function that updates the current context if one exists.
    
    Args:
        step_kind: Type of step - "llm", "tool", or None.
    """
    tracker = get_usage_tracker()
    if tracker is not None:
        tracker.record_step(step_kind)


def record_usage(tokens: int = 0, cost: float = 0.0) -> None:
    """Record usage in the current usage tracker.
    
    Convenience function that updates the current context if one exists.
    
    Args:
        tokens: Number of tokens used.
        cost: Cost in dollars.
    """
    tracker = get_usage_tracker()
    if tracker is not None:
        tracker.record_usage(tokens, cost)
