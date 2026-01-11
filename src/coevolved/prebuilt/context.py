"""Agent context management.

This module provides context variables for tracking agent execution state
(agent ID, type, run ID, iteration) without explicit parameter passing.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentContext:
    """Context information for agent execution.
    
    Stored in context variables to allow nested steps to access agent state
    without explicit parameter passing.
    
    Attributes:
        agent_id: Agent identifier.
        agent_type: Agent type (e.g., "react").
        agent_run_id: Unique identifier for this agent execution.
        iteration: Current iteration number in the agent loop.
    """
    agent_id: Optional[str]
    agent_type: Optional[str]
    agent_run_id: Optional[str]
    iteration: Optional[int]


_current_agent: ContextVar[Optional[AgentContext]] = ContextVar("current_agent", default=None)


def set_agent_context(ctx: AgentContext) -> None:
    """Set the current agent context in the context variable.
    
    Args:
        ctx: Agent context to set.
    """
    _current_agent.set(ctx)


def clear_agent_context() -> None:
    """Clear the current agent context."""
    _current_agent.set(None)


def get_agent_context() -> Optional[AgentContext]:
    """Get the current agent context.
    
    Returns:
        Current agent context, or None if not in an agent execution.
    """
    return _current_agent.get()
