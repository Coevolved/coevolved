"""Reusable agent loop infrastructure.

This module provides the agent_loop primitive for building agent patterns
with configurable iteration limits, checkpointing, and execution policies.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from coevolved.prebuilt.context import AgentContext, clear_agent_context, set_agent_context
from coevolved.prebuilt.events import AgentEvent
from coevolved.base.checkpoint import (
    Checkpoint,
    CheckpointPolicy,
    CheckpointStore,
    create_checkpoint,
)
from coevolved.base.interrupt import Interrupt
from coevolved.base.policy import (
    UsageTracker,
    UsagePolicy,
    clear_usage_tracker,
    get_usage_tracker,
    set_usage_tracker,
)
from coevolved.base.step import Step
from coevolved.base.tracing import Tracer, get_default_tracer


@dataclass
class LoopState:
    """Internal state for loop execution.
    
    Passed to the loop body function to provide iteration context.
    
    Attributes:
        iteration: Current iteration number (0-indexed).
        start_time: Unix timestamp when loop started.
        last_checkpoint_id: ID of the most recent checkpoint.
        agent_run_id: Unique ID for this loop execution.
    """
    iteration: int = 0
    start_time: float = field(default_factory=time.time)
    last_checkpoint_id: Optional[str] = None
    agent_run_id: Optional[str] = None
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time since loop started."""
        return time.time() - self.start_time


@dataclass
class LoopPolicy:
    """Configuration for agent loop execution.
    
    Attributes:
        max_iterations: Maximum number of loop iterations.
        timeout_seconds: Optional timeout for the entire loop.
        checkpoint_store: Optional store for checkpointing state.
        checkpoint_policy: When to create checkpoints.
        execution_policy: Optional policy for budget enforcement.
    
    Example:
        >>> policy = LoopPolicy(
        ...     max_iterations=10,
        ...     timeout_seconds=60.0,
        ...     checkpoint_store=MemoryCheckpointStore(),
        ... )
    """
    max_iterations: int = 10
    timeout_seconds: Optional[float] = None
    checkpoint_store: Optional[CheckpointStore] = None
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    execution_policy: Optional[UsagePolicy] = None


# Type aliases for clarity
StopCondition = Callable[[Any], bool]
LoopBody = Callable[[Any, LoopState], Any]


def agent_loop(
    *,
    body: LoopBody,
    stop_condition: StopCondition,
    policy: Optional[LoopPolicy] = None,
    name: str = "agent_loop",
    agent_type: str = "generic",
) -> Step:
    """Create a configurable agent loop as a Step.
    
    This extracts the loop pattern from react_agent into a reusable primitive.
    The loop repeatedly executes the body function until the stop condition
    is met or limits are reached.
    
    Args:
        body: Function that executes one iteration.
            Signature: (state, loop_state) -> new_state
        stop_condition: Function that returns True when loop should stop.
            Signature: (state) -> bool
        policy: Loop configuration (max iterations, checkpointing, etc.).
            Defaults to LoopPolicy() if not provided.
        name: Name for the agent (used in tracing).
        agent_type: Type identifier for tracing (e.g., "react", "plan_execute").
    
    Returns:
        Step that executes the loop.
    
    Example:
        >>> def react_body(state: dict, loop: LoopState) -> dict:
        ...     state = planner(state)
        ...     if state.get("tool_calls"):
        ...         state = execute_tools(state)
        ...     return state
        >>> 
        >>> react = agent_loop(
        ...     body=react_body,
        ...     stop_condition=lambda s: s.get("final") is not None,
        ...     policy=LoopPolicy(max_iterations=10),
        ...     name="react_agent",
        ...     agent_type="react",
        ... )
        >>> result = react(initial_state)
    """
    loop_policy = policy or LoopPolicy()
    
    def run(state: Any) -> Any:
        tracer = get_default_tracer()
        agent_run_id = f"{name}:{uuid.uuid4()}"
        loop_state = LoopState(agent_run_id=agent_run_id)
        
        # Set up usage tracking if policy provided
        outer_tracker = get_usage_tracker()
        if loop_policy.execution_policy:
            tracker = UsageTracker(loop_policy.execution_policy)
            set_usage_tracker(tracker)
        
        # Emit loop start event
        _emit_loop_event(
            tracer,
            agent_run_id,
            name,
            agent_type,
            "loop_start",
            max_iterations=loop_policy.max_iterations,
        )
        
        try:
            for iteration in range(loop_policy.max_iterations):
                loop_state.iteration = iteration
                
                # Check timeout
                if loop_policy.timeout_seconds:
                    if loop_state.elapsed_seconds > loop_policy.timeout_seconds:
                        _emit_loop_event(
                            tracer, agent_run_id, name, agent_type, "loop_timeout",
                            iteration=iteration,
                        )
                        break
                
                # Check execution policy
                if loop_policy.execution_policy:
                    tracker = get_usage_tracker()
                    if tracker and not tracker.check("step"):
                        break
                
                # Set agent context for nested steps
                agent_ctx = AgentContext(
                    agent_id=name,
                    agent_type=agent_type,
                    agent_run_id=agent_run_id,
                    iteration=iteration,
                )
                set_agent_context(agent_ctx)
                
                _emit_loop_event(
                    tracer, agent_run_id, name, agent_type, "iteration_start",
                    iteration=iteration,
                )
                
                try:
                    # Checkpoint before iteration if configured
                    if (loop_policy.checkpoint_store and 
                        loop_policy.checkpoint_policy.on_step_start):
                        cp = create_checkpoint(
                            run_id=tracer.run_id,
                            step_name=f"{name}:iter:{iteration}:start",
                            state=state,
                            parent_id=loop_state.last_checkpoint_id,
                        )
                        loop_policy.checkpoint_store.save(cp)
                        loop_state.last_checkpoint_id = cp.checkpoint_id
                    
                    # Execute body
                    state = body(state, loop_state)
                    
                    # Checkpoint after iteration if configured
                    if (loop_policy.checkpoint_store and 
                        loop_policy.checkpoint_policy.on_step_end):
                        cp = create_checkpoint(
                            run_id=tracer.run_id,
                            step_name=f"{name}:iter:{iteration}:end",
                            state=state,
                            parent_id=loop_state.last_checkpoint_id,
                        )
                        loop_policy.checkpoint_store.save(cp)
                        loop_state.last_checkpoint_id = cp.checkpoint_id
                    
                    # Record step in execution context
                    if loop_policy.execution_policy:
                        tracker = get_usage_tracker()
                        if tracker:
                            tracker.record_step("agent")
                    
                    _emit_loop_event(
                        tracer, agent_run_id, name, agent_type, "iteration_end",
                        iteration=iteration,
                    )
                    
                    # Check stop condition
                    if stop_condition(state):
                        _emit_loop_event(
                            tracer, agent_run_id, name, agent_type, "loop_end",
                            iteration=iteration,
                        )
                        return state
                    
                except Interrupt as e:
                    # Handle interrupt - checkpoint and re-raise
                    if (loop_policy.checkpoint_store and 
                        loop_policy.checkpoint_policy.on_interrupt):
                        cp = create_checkpoint(
                            run_id=tracer.run_id,
                            step_name=f"{name}:interrupt:{iteration}",
                            state=state,
                            parent_id=loop_state.last_checkpoint_id,
                            tags={"interrupt_id": e.interrupt_id},
                        )
                        loop_policy.checkpoint_store.save(cp)
                    raise
                
                except Exception as e:
                    # Handle error - checkpoint if configured
                    if (loop_policy.checkpoint_store and 
                        loop_policy.checkpoint_policy.on_error):
                        cp = create_checkpoint(
                            run_id=tracer.run_id,
                            step_name=f"{name}:error:{iteration}",
                            state=state,
                            parent_id=loop_state.last_checkpoint_id,
                            tags={"error": str(e)},
                        )
                        loop_policy.checkpoint_store.save(cp)
                    raise
            
            # Max iterations reached
            _emit_loop_event(
                tracer, agent_run_id, name, agent_type, "loop_max_iterations",
                iteration=loop_policy.max_iterations,
            )
            return state
            
        finally:
            clear_agent_context()
            # Restore outer usage tracker
            if outer_tracker:
                set_usage_tracker(outer_tracker)
            elif loop_policy.execution_policy:
                clear_usage_tracker()
    
    return Step(
        run,
        name=name,
        annotations={"kind": "agent", "agent_type": agent_type},
    )


def _emit_loop_event(
    tracer: Tracer,
    agent_run_id: str,
    name: str,
    agent_type: str,
    phase: str,
    *,
    iteration: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> None:
    """Emit an agent loop event for tracing."""
    tracer.emit(AgentEvent(
        run_id=tracer.run_id,
        step_id=None,
        invocation_id=None,
        group_hash=None,
        step_name=name,
        event=AgentEvent.EVENT_AGENT,
        timestamp=time.time(),
        agent_id=name,
        agent_type=agent_type,
        agent_run_id=agent_run_id,
        iteration=iteration,
        phase=phase,
        max_iterations=max_iterations,
    ))


def simple_loop(
    body: Callable[[Any], Any],
    stop_condition: StopCondition,
    *,
    max_iterations: int = 10,
    name: str = "loop",
) -> Step:
    """Create a simple loop without the full LoopState parameter.
    
    Convenience wrapper for agent_loop when you don't need loop state access.
    
    Args:
        body: Function that executes one iteration. Signature: (state) -> new_state
        stop_condition: Function that returns True when loop should stop.
        max_iterations: Maximum number of iterations.
        name: Name for the step.
    
    Returns:
        Step that executes the loop.
    
    Example:
        >>> loop_step = simple_loop(
        ...     body=lambda s: {**s, "count": s["count"] + 1},
        ...     stop_condition=lambda s: s["count"] >= 5,
        ... )
        >>> result = loop_step({"count": 0})  # {"count": 5}
    """
    def wrapped_body(state: Any, loop_state: LoopState) -> Any:
        return body(state)
    
    return agent_loop(
        body=wrapped_body,
        stop_condition=stop_condition,
        policy=LoopPolicy(max_iterations=max_iterations),
        name=name,
        agent_type="simple_loop",
    )
