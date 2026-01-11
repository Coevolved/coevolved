"""Checkpointing primitives for state persistence and resume.

This module provides the infrastructure for saving and restoring workflow state,
enabling resume from failure, human-in-the-loop workflows, and time-travel debugging.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class CheckpointMetadata:
    """Lightweight checkpoint reference without full state.
    
    Attributes:
        checkpoint_id: Unique identifier for the checkpoint.
        run_id: Identifier for the execution run.
        step_name: Name of the step that created this checkpoint.
        timestamp: Unix timestamp when checkpoint was created.
        tags: Optional tags for categorization and querying.
    """
    checkpoint_id: str
    run_id: str
    step_name: str
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """Full checkpoint with state.
    
    Attributes:
        checkpoint_id: Unique identifier for the checkpoint.
        run_id: Identifier for the execution run.
        step_name: Name of the step that created this checkpoint.
        state: The serializable state at this point.
        timestamp: Unix timestamp when checkpoint was created.
        parent_id: ID of the previous checkpoint in the chain (for lineage).
        tags: Optional tags for categorization and querying.
    
    Example:
        >>> checkpoint = create_checkpoint(
        ...     run_id="run-123",
        ...     step_name="process_data",
        ...     state={"processed": True, "count": 42},
        ... )
        >>> store.save(checkpoint)
    """
    checkpoint_id: str
    run_id: str
    step_name: str
    state: Any
    timestamp: float
    parent_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def metadata(self) -> CheckpointMetadata:
        """Get lightweight metadata without the full state."""
        return CheckpointMetadata(
            checkpoint_id=self.checkpoint_id,
            run_id=self.run_id,
            step_name=self.step_name,
            timestamp=self.timestamp,
            tags=self.tags,
        )


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends.
    
    Implementations must provide save, load, list, and get_latest operations.
    See MemoryCheckpointStore for a reference implementation.
    """
    
    def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint to storage.
        
        Args:
            checkpoint: The checkpoint to save.
        
        Returns:
            The checkpoint_id of the saved checkpoint.
        """
        ...
    
    def load(self, checkpoint_id: str) -> Checkpoint:
        """Load a checkpoint by ID.
        
        Args:
            checkpoint_id: The ID of the checkpoint to load.
        
        Returns:
            The loaded checkpoint.
        
        Raises:
            KeyError: If the checkpoint is not found.
        """
        ...
    
    def list_by_run(self, run_id: str) -> List[CheckpointMetadata]:
        """List all checkpoints for a run.
        
        Args:
            run_id: The run ID to query.
        
        Returns:
            List of checkpoint metadata, ordered by timestamp.
        """
        ...
    
    def get_latest(self, run_id: str) -> Optional[Checkpoint]:
        """Get the most recent checkpoint for a run.
        
        Args:
            run_id: The run ID to query.
        
        Returns:
            The latest checkpoint, or None if no checkpoints exist.
        """
        ...


class MemoryCheckpointStore:
    """In-memory checkpoint store for development and testing.
    
    Stores checkpoints in memory. Data is lost when the process exits.
    For production, implement a persistent CheckpointStore (e.g., Redis, PostgreSQL).
    
    Example:
        >>> store = MemoryCheckpointStore()
        >>> checkpoint = create_checkpoint("run-1", "step1", {"data": "value"})
        >>> store.save(checkpoint)
        >>> restored = store.get_latest("run-1")
    """
    
    def __init__(self) -> None:
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._by_run: Dict[str, List[str]] = {}
    
    def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint to memory."""
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        self._by_run.setdefault(checkpoint.run_id, []).append(checkpoint.checkpoint_id)
        return checkpoint.checkpoint_id
    
    def load(self, checkpoint_id: str) -> Checkpoint:
        """Load a checkpoint by ID."""
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Checkpoint '{checkpoint_id}' not found")
        return self._checkpoints[checkpoint_id]
    
    def list_by_run(self, run_id: str) -> List[CheckpointMetadata]:
        """List all checkpoints for a run, ordered by timestamp."""
        ids = self._by_run.get(run_id, [])
        checkpoints = [self._checkpoints[cid] for cid in ids]
        checkpoints.sort(key=lambda c: c.timestamp)
        return [c.metadata for c in checkpoints]
    
    def get_latest(self, run_id: str) -> Optional[Checkpoint]:
        """Get the most recent checkpoint for a run."""
        ids = self._by_run.get(run_id, [])
        if not ids:
            return None
        # Get all and sort by timestamp to find latest
        checkpoints = [self._checkpoints[cid] for cid in ids]
        checkpoints.sort(key=lambda c: c.timestamp)
        return checkpoints[-1]
    
    def clear(self) -> None:
        """Clear all checkpoints (useful for testing)."""
        self._checkpoints.clear()
        self._by_run.clear()


@dataclass
class CheckpointPolicy:
    """Configuration for when to create checkpoints.
    
    Attributes:
        on_step_start: Create checkpoint before step execution.
        on_step_end: Create checkpoint after successful step execution.
        on_error: Create checkpoint when an error occurs.
        on_interrupt: Create checkpoint when an interrupt is raised.
    
    Example:
        >>> # Checkpoint only on errors and interrupts (minimal overhead)
        >>> policy = CheckpointPolicy(on_step_end=False)
        >>> # Checkpoint everything (maximum recoverability)
        >>> policy = CheckpointPolicy(on_step_start=True, on_step_end=True)
    """
    on_step_start: bool = False
    on_step_end: bool = True
    on_error: bool = True
    on_interrupt: bool = True


def create_checkpoint(
    run_id: str,
    step_name: str,
    state: Any,
    *,
    parent_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Checkpoint:
    """Create a new checkpoint.
    
    Args:
        run_id: Identifier for the execution run.
        step_name: Name of the step creating this checkpoint.
        state: The state to checkpoint (must be serializable).
        parent_id: Optional ID of the parent checkpoint for lineage tracking.
        tags: Optional tags for categorization.
    
    Returns:
        A new Checkpoint instance with a unique ID and current timestamp.
    
    Example:
        >>> checkpoint = create_checkpoint(
        ...     run_id="run-abc",
        ...     step_name="extract_data",
        ...     state={"records": 100, "errors": 0},
        ...     tags={"phase": "extraction"},
        ... )
    """
    return Checkpoint(
        checkpoint_id=str(uuid.uuid4()),
        run_id=run_id,
        step_name=step_name,
        state=state,
        timestamp=datetime.now().timestamp(),
        parent_id=parent_id,
        tags=tags or {},
    )


def serialize_state(state: Any) -> str:
    """Serialize state to JSON string.
    
    Handles Pydantic models and common Python types.
    
    Args:
        state: State to serialize.
    
    Returns:
        JSON string representation.
    """
    def default_serializer(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)
    
    return json.dumps(state, default=default_serializer, sort_keys=True)


def deserialize_state(data: str) -> Any:
    """Deserialize state from JSON string.
    
    Args:
        data: JSON string to deserialize.
    
    Returns:
        Deserialized state (typically a dict).
    """
    return json.loads(data)
