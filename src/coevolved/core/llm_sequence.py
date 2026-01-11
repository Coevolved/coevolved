"""LLM sequence composition for chaining LLM Steps.

This module provides LLMSequence for creating deterministic chains of LLM Steps
that can be compiled for visualization or runtime optimization.
"""

from typing import Any, Optional, Sequence

from coevolved.base.compose import compile_steps, run_sequence
from coevolved.base.step import Step
from coevolved.base.tracing import Tracer


class LLMSequence:
    """Deterministic LLM Step chain that can be compiled for runtime optimization.
    
    LLMSequence enforces that all steps are LLM steps (kind="llm") and provides
    a compile() method for generating graph representations.
    
    Args:
        steps: Sequence of LLM Steps to chain.
        name: Optional name for the sequence.
    
    Attributes:
        steps: List of LLM Steps in the sequence.
        name: Sequence name.
    
    Example:
        >>> planner1 = llm_step(...)
        >>> planner2 = llm_step(...)
        >>> sequence = LLMSequence([planner1, planner2], name="two_stage")
        >>> result = sequence(state)
        >>> graph = sequence.compile()  # For visualization
    """

    def __init__(self, steps: Sequence[Step], name: Optional[str] = None) -> None:
        self.steps = list(steps)
        self.name = name or "llm_sequence"
        self._validate_steps()

    def __call__(self, state: Any, tracer: Optional[Tracer] = None) -> Any:
        """Execute the sequence of LLM steps sequentially.
        
        Args:
            state: Initial state to pass to the first step.
            tracer: Optional tracer instance.
        
        Returns:
            Final state after all steps have executed.
        """
        return run_sequence(self.steps, state, tracer=tracer)

    def compile(self) -> dict:
        """Compile the sequence into a graph representation.
        
        Returns:
            Dictionary with nodes, edges, entry, and exit information.
        """
        return compile_steps(self.steps, name=self.name)

    def _validate_steps(self) -> None:
        for step in self.steps:
            if step.annotations.get("kind") != "llm":
                raise ValueError(
                    f"LLMSequence can only chain LLM Steps; '{step.name}' has kind={step.annotations.get('kind')!r}."
                )


def llm_sequence(steps: Sequence[Step], name: Optional[str] = None) -> LLMSequence:
    """Create an LLMSequence from a sequence of Steps.
    
    Convenience function for creating LLMSequence instances.
    
    Args:
        steps: Sequence of LLM Steps to chain.
        name: Optional name for the sequence.
    
    Returns:
        LLMSequence instance.
    """
    return LLMSequence(steps, name=name)
