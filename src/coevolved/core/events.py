"""Event types for LLM and tool tracing.

This module defines event dataclasses for tracking LLM prompts, responses,
and tool executions.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

from coevolved.base.tracing import BaseEvent


@dataclass
class PromptEvent(BaseEvent):
    """Event emitted when a prompt is sent to an LLM.
    
    Attributes:
        prompt_id: Optional prompt identifier.
        prompt_version: Optional prompt version string.
        prompt_hash: Hash of the prompt content.
        prompt_text: Full prompt text (if log_prompt=True).
        annotations: Step metadata annotations.
        agent_id: Optional agent identifier (if in agent context).
        agent_type: Optional agent type (if in agent context).
        agent_run_id: Optional agent run identifier.
        iteration: Optional iteration number (if in agent loop).
    """
    EVENT_PROMPT: ClassVar[str] = "prompt"

    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None
    prompt_hash: Optional[str] = None
    prompt_text: Optional[str] = None
    annotations: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    agent_run_id: Optional[str] = None
    iteration: Optional[int] = None


@dataclass
class LLMEvent(BaseEvent):
    """Event emitted when an LLM response is received.
    
    Attributes:
        text: Generated text content.
        tool_calls: List of tool calls requested by the LLM.
        finish_reason: Reason for completion ("stop", "length", "tool_calls", etc.).
        model: Model identifier used.
        usage: Token usage statistics.
    """
    EVENT_RESPONSE: ClassVar[str] = "response"

    text: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


@dataclass
class ToolEvent(BaseEvent):
    """Event emitted when a tool is executed.
    
    Attributes:
        tool_name: Name of the tool.
        tool_args: Arguments passed to the tool.
        tool_result: Result returned by the tool.
        error: Error message (for "error" events).
    """
    EVENT_RESULT: ClassVar[str] = "result"
    EVENT_ERROR: ClassVar[str] = "error"

    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    error: Optional[str] = None
