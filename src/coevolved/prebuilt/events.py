"""Event types for agent tracing.

This module defines event dataclasses for tracking agent execution lifecycle.
"""

from dataclasses import dataclass
from typing import ClassVar, Optional

from coevolved.base.tracing import BaseEvent


@dataclass
class AgentEvent(BaseEvent):
    """Event emitted for agent execution lifecycle.
    
    Attributes:
        agent_id: Agent identifier.
        agent_type: Agent type (e.g., "react").
        agent_run_id: Unique identifier for this agent execution.
        iteration: Current iteration number in the agent loop.
        phase: Event phase ("loop_start", "iteration_start", "iteration_end", "loop_end").
        max_iterations: Maximum number of iterations (for "loop_start" events).
    """
    EVENT_AGENT: ClassVar[str] = "agent"

    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_run_id: Optional[str] = None
    iteration: Optional[int] = None
    phase: Optional[str] = None
    max_iterations: Optional[int] = None
