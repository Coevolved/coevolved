"""Composition utilities for combining Steps.

This module provides helpers for executing Steps in various patterns:
sequential, parallel, fallback chains, and with retry logic.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from coevolved.base.step import Step
from coevolved.base.tracing import Tracer, get_default_tracer


def run_sequence(steps: Sequence[Step], state: Any, tracer: Optional[Tracer] = None) -> Any:
    """Execute steps sequentially, passing output of each as input to the next.
    
    Args:
        steps: Sequence of steps to execute in order.
        state: Initial state to pass to the first step.
        tracer: Optional tracer instance. Uses default tracer if not provided.
    
    Returns:
        Final state after all steps have executed.
    
    Example:
        >>> step1 = Step(lambda s: {"x": s["a"] + 1})
        >>> step2 = Step(lambda s: {"result": s["x"] * 2})
        >>> result = run_sequence([step1, step2], {"a": 5})
        >>> result["result"]  # 12
    """
    tracer = tracer or get_default_tracer()
    current = state
    for step in steps:
        current = step(current, tracer=tracer)
    return current


def run_first_success(steps: Sequence[Step], state: Any, tracer: Optional[Tracer] = None) -> Any:
    """Execute steps until one succeeds (fallback chain pattern).
    
    Tries each step in sequence. Returns the first successful result.
    If all steps fail, raises the last exception.
    
    Args:
        steps: Sequence of steps to try in order.
        state: State to pass to each step.
        tracer: Optional tracer instance. Uses default tracer if not provided.
    
    Returns:
        Result from the first successful step.
    
    Raises:
        Exception: The last exception if all steps fail.
    
    Example:
        >>> primary = Step(lambda s: expensive_operation(s))
        >>> fallback = Step(lambda s: simple_operation(s))
        >>> result = run_first_success([primary, fallback], state)
    """
    tracer = tracer or get_default_tracer()
    last_error: Optional[Exception] = None
    for step in steps:
        try:
            return step(state, tracer=tracer)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error:
        raise last_error
    return state


@dataclass
class RetryPolicy:
    """Policy for retrying failed step executions.
    
    Attributes:
        attempts: Maximum number of attempts (default: 3).
        backoff_seconds: Initial delay between retries in seconds (default: 0.5).
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0).
        retry_exceptions: Tuple of exception types to retry on (default: all exceptions).
    
    Example:
        >>> policy = RetryPolicy(attempts=5, backoff_seconds=1.0)
        >>> retry_step = with_retry(step, policy)
    """
    attempts: int = 3
    backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    retry_exceptions: Tuple[type[BaseException], ...] = (Exception,)


def with_retry(step: Step, policy: RetryPolicy) -> Step:
    """Wrap a step with retry logic.
    
    Creates a new Step that retries the original step according to the
    retry policy. Uses exponential backoff between retries.
    
    Args:
        step: Step to wrap with retry logic.
        policy: Retry policy configuration.
    
    Returns:
        New Step with retry wrapper. The step name will be "{original_name}_retry".
    
    Example:
        >>> policy = RetryPolicy(attempts=3, backoff_seconds=1.0)
        >>> retry_step = with_retry(unreliable_step, policy)
        >>> result = retry_step(state)
    """
    def runner(state: Any) -> Any:
        delay = policy.backoff_seconds
        for attempt in range(1, policy.attempts + 1):
            try:
                return step(state)
            except policy.retry_exceptions:
                if attempt == policy.attempts:
                    raise
                time.sleep(delay)
                delay *= policy.backoff_multiplier
        return state

    return Step(
        runner,
        name=f"{step.name}_retry",
        input_schema=step.input_schema,
        output_schema=step.output_schema,
        annotations={**step.annotations, "retry_wrapper": True},
    )


def run_parallel(
    steps: Sequence[Step],
    state: Any,
    tracer: Optional[Tracer] = None,
    max_workers: Optional[int] = None,
) -> List[Any]:
    """Execute steps in parallel using a thread pool.
    
    All steps receive the same input state. Results are returned in the
    same order as the input steps list.
    
    Warning: Only use with steps that are safe to run in parallel (no
    shared mutable state, thread-safe operations).
    
    Args:
        steps: Sequence of steps to execute in parallel.
        state: State to pass to each step (same state for all).
        tracer: Optional tracer instance. Uses default tracer if not provided.
        max_workers: Maximum number of worker threads. Defaults to len(steps).
    
    Returns:
        List of results, one per step, in the same order as the input steps.
    
    Example:
        >>> results = run_parallel([step1, step2, step3], state)
        >>> # results[0] is from step1, results[1] from step2, etc.
    """
    tracer = tracer or get_default_tracer()
    results: List[Any] = [None] * len(steps)
    with ThreadPoolExecutor(max_workers=max_workers or len(steps)) as executor:
        future_map = {executor.submit(step, state, tracer): idx for idx, step in enumerate(steps)}
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
    return results


def compile_steps(steps: Sequence[Step], name: Optional[str] = None) -> Dict[str, Any]:
    """Compile a sequence of steps into a graph representation.
    
    Creates a graph structure with nodes (steps) and edges (sequential connections)
    that can be used for visualization or runtime optimization.
    
    Args:
        steps: Sequence of steps to compile.
        name: Optional name for the compiled graph.
    
    Returns:
        Dictionary with keys:
        - "name": Graph name
        - "entry": ID of the entry node
        - "exit": ID of the exit node
        - "nodes": List of node dictionaries with step metadata
        - "edges": List of edge dictionaries with "from" and "to" node IDs
    
    Example:
        >>> graph = compile_steps([step1, step2, step3], name="my_workflow")
        >>> print(graph["nodes"])  # List of step metadata
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []
    node_ids: List[str] = []

    for idx, step in enumerate(steps):
        node_id = f"{step.step_id}:{idx}"
        node_ids.append(node_id)
        node = {
            "id": node_id,
            "step_id": step.step_id,
            "name": step.name,
            "kind": step.annotations.get("kind", "step"),
            "annotations": step.annotations,
            "input_schema": _schema_name(step.input_schema),
            "output_schema": _schema_name(step.output_schema),
            "position": idx,
        }
        if step.annotations.get("kind") == "agent":
            node["agent_type"] = step.annotations.get("agent_type")
        nodes.append(node)

    for i in range(len(node_ids) - 1):
        edges.append({"from": node_ids[i], "to": node_ids[i + 1]})

    return {
        "name": name or "step_series",
        "entry": node_ids[0] if node_ids else None,
        "exit": node_ids[-1] if node_ids else None,
        "nodes": nodes,
        "edges": edges,
    }


def _schema_name(schema: Any) -> Optional[str]:
    return getattr(schema, "__name__", None) if schema else None
