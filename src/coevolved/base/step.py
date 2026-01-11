"""Core Step primitive for the agentic framework.

This module provides the Step class, which is the atomic unit of computation
in the framework. Steps wrap callables with optional typing, validation, and
automatic tracing.
"""

import hashlib
import inspect
import time
import uuid_utils as uuid
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from coevolved.base.tracing import (
    InvocationMetadata,
    SnapshotPolicy,
    StepEvent,
    Tracer,
    clear_current_invocation,
    get_default_tracer,
    set_current_invocation,
)

I = TypeVar("I")
O = TypeVar("O")
Schema = Optional[Type[BaseModel]]


def _to_dict(obj: Any) -> Any:
    """Convert a Pydantic model to a dictionary, or return the object as-is.
    
    Args:
        obj: Object to convert (Pydantic model or any other type).
        
    Returns:
        Dictionary representation if obj is a BaseModel, otherwise obj unchanged.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj


class Step(Generic[I, O]):
    """Atomic unit of computation in the framework.
    
    A Step wraps any callable with optional type validation, automatic tracing,
    and metadata annotations. Steps are the fundamental building blocks for
    composing agentic workflows.
    
    Steps execute with automatic:
    - Input/output validation (if schemas provided)
    - Tracing events (start, end, error)
    - Invocation context tracking
    - Snapshot capture (configurable)
    
    Args:
        fn: The callable function to wrap. Must accept one argument (state).
        name: Optional name for the step. Defaults to function name.
        input_schema: Optional Pydantic model for input validation.
        output_schema: Optional Pydantic model for output validation.
        annotations: Optional dictionary of metadata for runtime optimizations.
            Common keys: "kind" (e.g., "llm", "tool", "agent"), "cacheable", etc.
    
    Attributes:
        fn: The wrapped callable.
        name: Step name.
        input_schema: Input validation schema.
        output_schema: Output validation schema.
        annotations: Metadata dictionary.
        step_id: Unique identifier computed from function source and metadata.
    
    Example:
        >>> def process_data(state: dict) -> dict:
        ...     return {"processed": state["data"]}
        >>> step = Step(process_data, name="process")
        >>> result = step({"data": "input"})
    """
    
    def __init__(
        self,
        fn: Callable[[I], O],
        name: Optional[str] = None,
        input_schema: Schema = None,
        output_schema: Schema = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "step")
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.annotations = dict(annotations or {})
        self.step_id = self._compute_step_id()

    def __call__(self, state: I, tracer: Optional[Tracer] = None) -> O:
        """Execute the step with the given state.
        
        Args:
            state: Input state (will be validated if input_schema is set).
            tracer: Optional tracer instance. Uses default tracer if not provided.
        
        Returns:
            Output state (validated if output_schema is set).
        
        Raises:
            ValueError: If input or output validation fails.
            Exception: Any exception raised by the wrapped function is re-raised
                after emitting an error event.
        """
        tracer = tracer or get_default_tracer()
        snapshot_policy = getattr(tracer, "snapshot_policy", SnapshotPolicy())
        start = time.perf_counter()
        validated_input = self._validate_input(state)
        input_hash, input_hash_is_repr = tracer.input_hash(validated_input)
        group_hash = _group_hash(tracer.run_id, self.name, input_hash)
        invocation_id = _new_invocation_id()
        set_current_invocation(
            InvocationMetadata(
                run_id=tracer.run_id,
                step_id=self.step_id,
                invocation_id=invocation_id,
                group_hash=group_hash,
                step_name=self.name,
                input_hash=input_hash,
                input_hash_is_repr=input_hash_is_repr,
            )
        )
        input_snapshot = (
            self._snapshot(validated_input)
            if snapshot_policy.include_input(StepEvent.EVENT_START)
            else None
        )
        tracer.emit(
            StepEvent(
                run_id=tracer.run_id,
                step_id=self.step_id,
                invocation_id=invocation_id,
                group_hash=group_hash,
                step_name=self.name,
                event=StepEvent.EVENT_START,
                timestamp=time.time(),
                input_hash=input_hash,
                input_hash_is_repr=input_hash_is_repr,
                annotations=self.annotations or None,
                input_snapshot=input_snapshot,
            )
        )
        try:
            result = self.fn(validated_input)
            validated_output = self._validate_output(result)
            output_snapshot = (
                self._snapshot(validated_output)
                if snapshot_policy.include_output(StepEvent.EVENT_END)
                else None
            )
            tracer.emit(
                StepEvent(
                    run_id=tracer.run_id,
                    step_id=self.step_id,
                    invocation_id=invocation_id,
                    group_hash=group_hash,
                    step_name=self.name,
                    event=StepEvent.EVENT_END,
                    timestamp=time.time(),
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    annotations=self.annotations or None,
                    output_snapshot=output_snapshot,
                )
            )
            return validated_output
        except Exception as exc:
            error_snapshot = input_snapshot
            if error_snapshot is None and snapshot_policy.include_input(StepEvent.EVENT_ERROR):
                error_snapshot = self._snapshot(validated_input)
            tracer.emit(
                StepEvent(
                    run_id=tracer.run_id,
                    step_id=self.step_id,
                    invocation_id=invocation_id,
                    group_hash=group_hash,
                    step_name=self.name,
                    event=StepEvent.EVENT_ERROR,
                    timestamp=time.time(),
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    input_hash=input_hash,
                    input_hash_is_repr=input_hash_is_repr,
                    annotations=self.annotations or None,
                    input_snapshot=error_snapshot,
                    error=str(exc),
                )
            )
            raise
        finally:
            clear_current_invocation()

    def _validate_input(self, state: Any) -> Any:
        """Validate input state against input_schema if set.
        
        Args:
            state: Input state to validate.
        
        Returns:
            Validated state (Pydantic model instance if schema provided).
        
        Raises:
            ValueError: If validation fails.
        """
        if self.input_schema:
            try:
                validated = self.input_schema.model_validate(state)
            except ValidationError as exc:
                raise ValueError(f"Input validation failed for step '{self.name}': {exc}") from exc
            return validated  # type: ignore[return-value]
        return state

    def _validate_output(self, result: Any) -> Any:
        """Validate output result against output_schema if set.
        
        Args:
            result: Output result to validate.
        
        Returns:
            Validated result (Pydantic model instance if schema provided).
        
        Raises:
            ValueError: If validation fails.
        """
        if self.output_schema:
            try:
                validated = self.output_schema.model_validate(result)
            except ValidationError as exc:
                raise ValueError(f"Output validation failed for step '{self.name}': {exc}") from exc
            return validated  # type: ignore[return-value]
        return result

    def _snapshot(self, payload: Any) -> Optional[Dict[str, Any]]:
        """Create a snapshot dictionary from payload for tracing.
        
        Args:
            payload: Data to snapshot (typically input or output).
        
        Returns:
            Dictionary representation, or None if serialization fails.
        """
        try:
            data = _to_dict(payload)
            return data if isinstance(data, dict) else {"value": data}
        except Exception:
            # Avoid trace failures due to serialization issues.
            return None

    def __repr__(self) -> str:
        """Return string representation of the step."""
        return f"<Step name={self.name}>"

    def _compute_step_id(self) -> str:
        """Compute a unique identifier for this step.
        
        The ID is based on the function's module, qualname, schemas, annotations,
        and source code, ensuring uniqueness while allowing the same logical step
        to have the same ID across invocations.
        
        Returns:
            12-character hexadecimal hash of the step's signature.
        """
        module = getattr(self.fn, "__module__", "")
        qualname = getattr(self.fn, "__qualname__", self.name)
        schema_sig = (
            getattr(self.input_schema, "__name__", None),
            getattr(self.output_schema, "__name__", None),
        )
        ann_fingerprint = hashlib.sha256(repr(sorted(self.annotations.items())).encode("utf-8")).hexdigest()
        try:
            source = inspect.getsource(self.fn)
        except OSError:
            source = qualname
        blob = f"{module}:{qualname}:{schema_sig}:{ann_fingerprint}:{source}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def step(
    *,
    name: Optional[str] = None,
    input_schema: Schema = None,
    output_schema: Schema = None,
    annotations: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[[I], O]], Step[I, O]]:
    """Decorator for creating Steps with metadata.
    
    Args:
        name: Optional name for the step. Defaults to function name.
        input_schema: Optional Pydantic model for input validation.
        output_schema: Optional Pydantic model for output validation.
        annotations: Optional dictionary of metadata.
    
    Returns:
        Decorator function that converts a callable into a Step.
    
    Example:
        >>> @step(name="process", input_schema=InputModel)
        ... def process_data(state: InputModel) -> OutputModel:
        ...     return OutputModel(result=state.data)
    """

    def wrapper(fn: Callable[[I], O]) -> Step[I, O]:
        return Step(
            fn,
            name=name,
            input_schema=input_schema,
            output_schema=output_schema,
            annotations=annotations,
        )

    return wrapper


def _invocation_hash(run_id: str, step_name: str, input_hash: str) -> str:
    """Compute a hash for a step invocation.
    
    Args:
        run_id: Unique run identifier.
        step_name: Name of the step.
        input_hash: Hash of the input state.
    
    Returns:
        12-character hexadecimal hash.
    """
    raw = f"{run_id}:{step_name}:{input_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _group_hash(run_id: str, step_name: str, input_hash: str) -> str:
    """Compute a group hash for grouping related invocations.
    
    Args:
        run_id: Unique run identifier.
        step_name: Name of the step.
        input_hash: Hash of the input state.
    
    Returns:
        12-character hexadecimal hash.
    """
    return _invocation_hash(run_id, step_name, input_hash)


def _new_invocation_id() -> str:
    """Generate a unique invocation identifier.
    
    Returns:
        A globally-unique identifier (UUID hex, 32 chars).
    """
    # Don't truncate: we want collision resistance across processes/threads.
    return uuid.uuid7().hex
