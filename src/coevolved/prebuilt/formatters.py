"""Formatter for agent trace events.

This module provides formatters that create readable string representations
of agent events for console logging.
"""

from typing import Optional

from coevolved.base.tracing import BaseEvent
from coevolved.prebuilt.events import AgentEvent


class DefaultAgentFormatter:
    """Formatter for agent events.
    
    Creates compact log lines showing agent phase, ID, type, and iteration.
    """
    def format(self, event: BaseEvent) -> Optional[str]:
        if not isinstance(event, AgentEvent):
            return None
        phase = event.phase or "agent"
        parts = [f"Agent::{phase}"]
        agent_id = event.agent_id or event.step_name
        if agent_id:
            parts.append(f"agent={agent_id}")
        if event.agent_type:
            parts.append(f"type={event.agent_type}")
        if event.iteration is not None:
            parts.append(f"iter={event.iteration}")
        if event.max_iterations is not None and phase == "loop_start":
            parts.append(f"max_iter={event.max_iterations}")
        return " ".join(parts)
