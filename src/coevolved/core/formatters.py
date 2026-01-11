"""Formatters for LLM and tool trace events.

This module provides formatters that create readable string representations
of LLM-related events for console logging.
"""

import json
from typing import Any, Dict, Optional

from coevolved.base.tracing import BaseEvent, StepEvent
from coevolved.core.events import LLMEvent, PromptEvent, ToolEvent


class DefaultLLMFormatter:
    """Formatter for LLM, tool, and prompt events.
    
    Creates compact, readable log lines with truncated text and arguments
    to keep output manageable.
    
    Args:
        max_text_chars: Maximum characters for text fields (default: 160).
        max_args_chars: Maximum characters for argument fields (default: 120).
    """
    def __init__(self, *, max_text_chars: int = 160, max_args_chars: int = 120) -> None:
        self.max_text_chars = max_text_chars
        self.max_args_chars = max_args_chars

    def format(self, event: BaseEvent) -> Optional[str]:
        if isinstance(event, PromptEvent):
            return self._format_prompt(event)
        if isinstance(event, LLMEvent):
            return self._format_llm_event(event)
        if isinstance(event, ToolEvent):
            return self._format_tool_event(event)
        if isinstance(event, StepEvent):
            return self._format_step_event(event)
        return None

    def _format_prompt(self, event: PromptEvent) -> str:
        parts = ["LLM::prompt", f"step={event.step_name}"]
        if event.prompt_id:
            parts.append(f"prompt_id={event.prompt_id}")
        if event.prompt_version:
            parts.append(f"prompt_version={event.prompt_version}")
        if event.prompt_hash:
            parts.append(f"prompt_hash={event.prompt_hash}")
        if event.prompt_text:
            preview = _truncate(_one_line(event.prompt_text), self.max_text_chars)
            parts.append(f"text={preview}")
        return " ".join(parts)

    def _format_llm_event(self, event: LLMEvent) -> str:
        parts = ["LLM::output", f"step={event.step_name}"]
        tool_calls = _coerce_tool_calls(event.tool_calls)
        if tool_calls:
            parts.append(f"tool_calls={_format_tool_calls(tool_calls, self.max_args_chars)}")
        if event.text:
            truncated_text = _truncate(_one_line(event.text), self.max_text_chars)
            parts.append(f"text=\"{truncated_text}\"")
        if event.finish_reason:
            parts.append(f"finish={event.finish_reason}")
        if event.model:
            parts.append(f"model={event.model}")
        parts.append(f"invocation_id={event.invocation_id}")
        return " ".join(parts)

    def _format_tool_event(self, event: ToolEvent) -> str:
        if event.event == "error":
            return f"Tool::error tool={event.tool_name or event.step_name} error={event.error}"
        parts = ["Tool::result", f"tool={event.tool_name or event.step_name}"]
        if event.tool_args is not None:
            parts.append(f"args={_truncate(_one_line(_stringify(event.tool_args)), self.max_args_chars)}")
        if event.tool_result is not None:
            parts.append(
                f"output={_truncate(_one_line(_stringify(event.tool_result)), self.max_text_chars)}"
            )
        return " ".join(parts)

    def _format_step_event(self, event: StepEvent) -> Optional[str]:
        if event.event == "start":
            return None
        label = _annotation_kind(event.annotations) or "step"
        label = {"llm": "LLM", "tool": "Tool", "agent": "Agent"}.get(
            label, label.capitalize()
        )
        if event.event == "error":
            line = f"{label}::error step={event.step_name} error={event.error}"
        else:
            line = f"{label}::end step={event.step_name}"
        if event.elapsed_ms is not None:
            line = f"{line} elapsed_ms={event.elapsed_ms:.2f}"
        return line


def _annotation_kind(annotations: Any) -> Optional[str]:
    if isinstance(annotations, dict):
        kind = annotations.get("kind")
        if kind:
            return str(kind)
    return None


def _coerce_tool_calls(value: Any) -> list[Dict[str, Any]]:
    if value is None:
        return []
    calls = []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    for item in items:
        if isinstance(item, dict):
            calls.append(item)
        else:
            calls.append({"value": str(item)})
    return calls


def _format_tool_calls(tool_calls: list[Dict[str, Any]], limit: int) -> str:
    rendered = []
    for call in tool_calls:
        name = call.get("name") or call.get("function", {}).get("name")
        args = call.get("arguments")
        if args is None:
            args = call.get("function", {}).get("arguments")
        if args is None:
            args = {}
        args_str = _truncate(_one_line(_stringify(args)), limit)
        rendered.append(f"{name}({args_str})")
    return f"[{', '.join(rendered)}]"


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _one_line(value: str) -> str:
    return " ".join(value.splitlines())
