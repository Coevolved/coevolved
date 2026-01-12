"""Interrupt handling for human-in-the-loop workflows.

This module provides the Interrupt exception and related utilities for pausing
workflow execution to request external input, then resuming with the provided value.
"""

import hashlib
import inspect
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class InterruptPayload:
    """Information surfaced when an interrupt occurs.
    
    Attributes:
        interrupt_id: Unique identifier for this interrupt.
        value: Context for the human (e.g., message, data to review).
        step_name: Name of the step that raised the interrupt.
        run_id: Identifier for the execution run.
    """
    interrupt_id: str
    value: Any
    step_name: str
    run_id: str


class Interrupt(Exception):
    """Exception raised to pause execution and request external input.
    
    When raised within a workflow, execution pauses and the interrupt value
    is surfaced to the caller. The workflow can be resumed by providing
    a response value.
    
    Attributes:
        value: Context or data to present to the human.
        interrupt_id: Unique identifier for this interrupt.
    
    Example:
        >>> def approval_step(state: dict) -> dict:
        ...     if state["amount"] > 1000:
        ...         # This will pause execution
        ...         approval = interrupt({
        ...             "message": "Large transaction requires approval",
        ...             "amount": state["amount"],
        ...         })
        ...         if not approval.get("approved"):
        ...             raise ValueError("Transaction rejected")
        ...     return state
    """
    
    def __init__(self, value: Any, *, interrupt_id: Optional[str] = None) -> None:
        self.value = value
        self.interrupt_id = interrupt_id or str(uuid.uuid4())
        super().__init__(f"Interrupt requested: {value}")
    
    def to_payload(self, step_name: str, run_id: str) -> InterruptPayload:
        """Convert to InterruptPayload with step/run context.
        
        Args:
            step_name: Name of the step that raised this interrupt.
            run_id: Identifier for the execution run.
        
        Returns:
            InterruptPayload with full context.
        """
        return InterruptPayload(
            interrupt_id=self.interrupt_id,
            value=self.value,
            step_name=step_name,
            run_id=run_id,
        )


# Context variables for resume handling
_resume_values: ContextVar[dict[str, Any]] = ContextVar("resume_values", default={})


def _default_interrupt_id() -> str:
    """Generate a stable interrupt_id for the current call site.

    We want a deterministic id so that callers can resume by providing a value
    for the same interrupt_id on a subsequent execution.
    """
    try:
        frame = inspect.currentframe()
        caller = frame.f_back if frame else None
        if caller is None:
            raise RuntimeError("No caller frame available")
        callsite = f"{caller.f_code.co_filename}:{caller.f_lineno}:{caller.f_code.co_name}"
        # Keep it reasonably short but collision-resistant for typical projects.
        return hashlib.sha256(callsite.encode("utf-8")).hexdigest()[:32]
    except Exception:
        # Fallback: still stable enough within a single run, and avoids failing.
        return str(uuid.uuid4())


def interrupt(value: Any, *, key: Optional[str] = None) -> Any:
    """Pause execution and request human input.
    
    On first call within a step, raises an Interrupt exception to pause execution.
    When the workflow is resumed with a value for this interrupt, the function
    returns that value instead of raising.
    
    Args:
        value: Context or data to present to the human. Can be any serializable
            value (string, dict, etc.).
        key: Optional stable key to use as the interrupt_id. Use this when the
            same call site may legitimately need distinct interrupts (e.g., inside
            a loop) and you want to disambiguate them.
    
    Returns:
        The value provided when resuming (only on resume, not first call).
    
    Raises:
        Interrupt: On first call, to pause execution.
    
    Example:
        >>> def review_step(state: dict) -> dict:
        ...     # First execution: raises Interrupt
        ...     # On resume: returns the human's response
        ...     response = interrupt({
        ...         "question": "Is this correct?",
        ...         "data": state["draft"],
        ...     })
        ...     state["approved"] = response.get("approved", False)
        ...     return state
    """
    interrupt_id = key or _default_interrupt_id()

    # Check if we have a resume value for this interrupt.
    # Always copy + set to avoid mutating the ContextVar default dict.
    resume_values = _resume_values.get().copy()
    if interrupt_id in resume_values:
        resumed = resume_values.pop(interrupt_id)
        _resume_values.set(resume_values)
        return resumed

    # No resume value - raise to pause execution.
    raise Interrupt(value, interrupt_id=interrupt_id)


def set_resume_value(interrupt_id: str, value: Any) -> None:
    """Set the value to return when a specific interrupt is encountered.
    
    Call this before re-executing a step to provide the human's response.
    
    Args:
        interrupt_id: The ID of the interrupt to resume.
        value: The value to return from the interrupt() call.
    """
    resume_values = _resume_values.get().copy()
    resume_values[interrupt_id] = value
    _resume_values.set(resume_values)


def set_resume_values(values: dict[str, Any]) -> None:
    """Set multiple resume values at once.
    
    Args:
        values: Dictionary mapping interrupt IDs to resume values.
    """
    current = _resume_values.get().copy()
    current.update(values)
    _resume_values.set(current)


def clear_resume_values() -> None:
    """Clear all pending resume values."""
    _resume_values.set({})


def get_pending_resume_values() -> dict[str, Any]:
    """Get all pending resume values (for debugging/inspection)."""
    return _resume_values.get().copy()


@dataclass
class RunResult:
    """Result of executing a workflow that may have been interrupted.
    
    Attributes:
        status: Execution status - "complete", "interrupted", or "error".
        value: Final result value (for "complete" status).
        interrupt: Interrupt payload (for "interrupted" status).
        error: Exception that occurred (for "error" status).
        checkpoint_id: ID of the last checkpoint (if checkpointing enabled).
    
    Example:
        >>> result = run_with_interrupts(workflow, state, store=store)
        >>> if result.status == "interrupted":
        ...     print(f"Needs input: {result.interrupt.value}")
        ...     human_response = get_user_input()
        ...     result = run_with_interrupts(
        ...         workflow, 
        ...         result.interrupt.state,
        ...         store=store,
        ...         resume={result.interrupt.interrupt_id: human_response},
        ...     )
    """
    status: Literal["complete", "interrupted", "error"]
    value: Any = None
    interrupt: Optional[InterruptPayload] = None
    error: Optional[Exception] = None
    checkpoint_id: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if execution completed successfully."""
        return self.status == "complete"
    
    @property
    def is_interrupted(self) -> bool:
        """Check if execution was interrupted."""
        return self.status == "interrupted"
    
    @property
    def is_error(self) -> bool:
        """Check if execution failed with an error."""
        return self.status == "error"


@dataclass
class InterruptContext:
    """Context for tracking interrupts within a workflow execution.
    
    Used internally by workflow runners to track interrupt state.
    """
    run_id: str
    step_name: str = ""
    interrupts_seen: list[str] = None
    
    def __post_init__(self):
        if self.interrupts_seen is None:
            self.interrupts_seen = []
    
    def record_interrupt(self, interrupt: Interrupt) -> None:
        """Record that an interrupt was raised."""
        self.interrupts_seen.append(interrupt.interrupt_id)


# Context variable for current interrupt context
_interrupt_context: ContextVar[Optional[InterruptContext]] = ContextVar(
    "interrupt_context", default=None
)


def get_interrupt_context() -> Optional[InterruptContext]:
    """Get the current interrupt context."""
    return _interrupt_context.get()


def set_interrupt_context(ctx: InterruptContext) -> None:
    """Set the current interrupt context."""
    _interrupt_context.set(ctx)


def clear_interrupt_context() -> None:
    """Clear the current interrupt context."""
    _interrupt_context.set(None)
