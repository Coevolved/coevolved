"""Tracing and observability system for the framework.

This module provides event-based tracing with pluggable sinks, formatters,
and snapshot policies for capturing execution metadata.
"""

import hashlib
import json
import logging
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Protocol, Tuple


@dataclass
class BaseEvent:
    """Minimal event payload shared across all event types.
    
    Attributes:
        run_id: Unique identifier for the execution run.
        step_id: Unique identifier for the step definition.
        invocation_id: Unique identifier for this specific invocation.
        group_hash: Hash for grouping related invocations.
        step_name: Human-readable step name.
        event: Event type (e.g., "start", "end", "error").
        timestamp: Unix timestamp of the event.
    """

    run_id: str
    step_id: Optional[str]
    invocation_id: Optional[str]
    group_hash: Optional[str]
    step_name: str
    event: str
    timestamp: float


@dataclass
class StepEvent(BaseEvent):
    """Event emitted for step execution lifecycle.
    
    Attributes:
        elapsed_ms: Execution time in milliseconds (for "end" and "error" events).
        input_hash: Hash of the input state.
        input_hash_is_repr: Whether the hash was computed from repr() fallback.
        annotations: Step metadata annotations.
        input_snapshot: Captured input state (if snapshot policy allows).
        output_snapshot: Captured output state (if snapshot policy allows).
        error: Error message (for "error" events).
    """
    EVENT_START: ClassVar[str] = "start"
    EVENT_END: ClassVar[str] = "end"
    EVENT_ERROR: ClassVar[str] = "error"

    elapsed_ms: Optional[float] = None
    input_hash: Optional[str] = None
    input_hash_is_repr: bool = False
    annotations: Optional[Dict[str, Any]] = None
    input_snapshot: Optional[Dict[str, Any]] = None
    output_snapshot: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class SnapshotPolicy:
    """Policy for capturing input/output snapshots in trace events.
    
    Controls when input and output data should be captured for debugging
    and observability. By default, only errors capture input snapshots.
    
    Attributes:
        capture_start: Whether to capture input snapshot on "start" events.
        capture_end: Whether to capture output snapshot on "end" events.
        capture_error: Whether to capture input snapshot on "error" events.
    
    Example:
        >>> # Only capture errors (default)
        >>> policy = SnapshotPolicy.errors_only()
        >>> # Capture everything
        >>> policy = SnapshotPolicy(True, True, True)
    """
    capture_start: bool = False
    capture_end: bool = False
    capture_error: bool = True

    @classmethod
    def none(cls) -> "SnapshotPolicy":
        """Create a policy that captures no snapshots."""
        return cls(False, False, False)

    @classmethod
    def errors_only(cls) -> "SnapshotPolicy":
        """Create a policy that only captures snapshots on errors (default)."""
        return cls(False, False, True)

    @classmethod
    def start_only(cls) -> "SnapshotPolicy":
        """Create a policy that only captures input on start."""
        return cls(True, False, False)

    @classmethod
    def end_only(cls) -> "SnapshotPolicy":
        """Create a policy that only captures output on end."""
        return cls(False, True, False)

    def include_input(self, event: str) -> bool:
        """Check if input snapshot should be captured for the given event.
        
        Args:
            event: Event type ("start", "end", "error").
        
        Returns:
            True if input should be captured.
        """
        if event == StepEvent.EVENT_START:
            return self.capture_start
        if event == StepEvent.EVENT_ERROR:
            return self.capture_error
        return False

    def include_output(self, event: str) -> bool:
        """Check if output snapshot should be captured for the given event.
        
        Args:
            event: Event type (typically "end").
        
        Returns:
            True if output should be captured.
        """
        return event == StepEvent.EVENT_END and self.capture_end


class TraceSink(Protocol):
    """Protocol for trace event sinks.
    
    Sinks receive events and handle them (e.g., logging, writing to file,
    sending to external service).
    """
    def emit(self, event: BaseEvent) -> None:
        """Emit a trace event.
        
        Args:
            event: The event to emit.
        """
        ...


class EventFormatter(Protocol):
    """Protocol for formatting trace events into strings."""
    def format(self, event: BaseEvent) -> Optional[str]:
        """Format an event into a string representation.
        
        Args:
            event: The event to format.
        
        Returns:
            Formatted string, or None to skip formatting.
        """
        ...


class DefaultFormatter:
    """Default formatter that creates a simple key=value representation."""
    def format(self, event: BaseEvent) -> Optional[str]:
        """Format event as key=value pairs.
        
        Args:
            event: Event to format.
        
        Returns:
            Formatted string representation.
        """
        payload = {k: v for k, v in asdict(event).items() if v is not None}
        payload["event_type"] = event.__class__.__name__
        return _format_event(payload)


class PerTypeFormatter:
    """Formatter that delegates to type-specific formatters.
    
    Allows different event types to be formatted differently while providing
    a default formatter for unhandled types.
    
    Args:
        default: Default formatter for unhandled event types.
        overrides: Dictionary mapping event types to their specific formatters.
    """
    def __init__(self, default: EventFormatter, overrides: Dict[type, EventFormatter]) -> None:
        self.default = default
        self.overrides = overrides

    def format(self, event: BaseEvent) -> Optional[str]:
        formatter = self.overrides.get(type(event), self.default)
        return formatter.format(event)


class ConsoleSink:
    """Lightweight default sink; logs a compact, readable line.
    
    Args:
        formatter: Optional formatter. Uses DefaultFormatter if not provided.
    """

    def __init__(self, formatter: Optional[EventFormatter] = None) -> None:
        self.formatter = formatter or DefaultFormatter()

    def emit(self, event: BaseEvent) -> None:
        logger = logging.getLogger("base.tracing")
        summary = self.formatter.format(event)
        if summary:
            logger.info(summary)


class Tracer:
    """Central tracing coordinator.
    
    Manages trace event emission, input hashing, and snapshot policies.
    Each tracer instance has a unique run_id for grouping related events.
    
    Args:
        sinks: List of sinks to emit events to. Defaults to empty list.
        run_id: Unique identifier for this run. Auto-generated if not provided.
        redactor: Optional function to redact sensitive data before hashing.
        snapshot_policy: Policy for capturing snapshots. Defaults to errors_only.
    
    Attributes:
        sinks: List of trace sinks.
        run_id: Unique run identifier.
        redactor: Function for redacting sensitive data.
        snapshot_policy: Policy for snapshot capture.
    """
    def __init__(
        self,
        sinks: Optional[List[TraceSink]] = None,
        run_id: Optional[str] = None,
        redactor: Optional[Callable[[Any], Any]] = None,
        snapshot_policy: Optional[SnapshotPolicy] = None,
    ) -> None:
        self.sinks = sinks or []
        self.run_id = run_id or str(uuid.uuid4())
        self.redactor = redactor
        self.snapshot_policy = snapshot_policy or SnapshotPolicy()

    def emit(self, event: BaseEvent) -> None:
        """Emit an event to all configured sinks.
        
        Failures in sinks are silently ignored to prevent tracing issues
        from breaking application logic.
        
        Args:
            event: Event to emit.
        """
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception:
                # Tracing failures must not break application logic.
                continue

    def input_hash(self, payload: Any) -> Tuple[str, bool]:
        """Compute a hash of the input payload.
        
        Args:
            payload: Input data to hash.
        
        Returns:
            Tuple of (hash_string, is_repr_fallback). The boolean indicates
            whether the hash was computed from repr() fallback (True) or
            JSON serialization (False).
        """
        try:
            scrubbed = self.redactor(payload) if self.redactor else payload
            canonical = json.dumps(scrubbed, sort_keys=True, default=_default_json_serializer)
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), False
        except Exception:
            fallback = repr(payload)
            return hashlib.sha256(fallback.encode("utf-8")).hexdigest(), True


_default_tracer: Tracer = Tracer([ConsoleSink()])


def get_default_tracer() -> Tracer:
    """Get the default tracer instance.
    
    Returns:
        The global default tracer (initialized with ConsoleSink).
    """
    return _default_tracer


def set_default_tracer(tracer: Tracer) -> None:
    """Set the default tracer instance.
    
    Args:
        tracer: Tracer to use as the default.
    """
    global _default_tracer
    _default_tracer = tracer


def _default_json_serializer(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _format_event(payload: Dict[str, Any]) -> str:
    parts = [
        f"{payload.get('event_type')}::{payload.get('event')}",
        f"step={payload.get('step_name')}",
    ]
    for key in payload:
        parts.append(f"{key}={payload[key]}")
    return " ".join(parts)


@dataclass
class InvocationMetadata:
    """Metadata information for the current step invocation.
    
    Stored in metadata variables to allow nested steps to access their
    execution metadata without explicit parameter passing.
    
    Attributes:
        run_id: Unique identifier for the execution run.
        step_id: Unique identifier for the step definition.
        invocation_id: Unique identifier for this specific invocation.
        group_hash: Hash for grouping related invocations.
        step_name: Human-readable step name.
        input_hash: Hash of the input state.
        input_hash_is_repr: Whether the hash was computed from repr() fallback.
    """
    run_id: str
    step_id: str
    invocation_id: str
    group_hash: str
    step_name: str
    input_hash: str
    input_hash_is_repr: bool


_current_invocation: ContextVar[Optional[InvocationMetadata]] = ContextVar(
    "current_invocation", default=None
)


def set_current_invocation(ctx: InvocationMetadata) -> None:
    """Set the current invocation context in the context variable.
    
    Args:
        ctx: Invocation context to set.
    """
    _current_invocation.set(ctx)


def clear_current_invocation() -> None:
    """Clear the current invocation context."""
    _current_invocation.set(None)


def get_current_invocation() -> Optional[InvocationMetadata]:
    """Get the current invocation context.
    
    Returns:
        Current invocation context, or None if not in a step execution.
    """
    return _current_invocation.get()


TraceEvent = StepEvent
