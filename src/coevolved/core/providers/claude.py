"""Claude (Anthropic) provider implementation for LLM requests.

This module provides a ClaudeProvider that implements both LLMProvider and
StreamingLLMProvider protocols for use with Anthropic's API client.
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

from coevolved.core.types import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    PromptPayload,
    ToolCall,
    ToolSpec,
)


class ClaudeProvider:
    """LLM provider implementation for Anthropic's Claude API.

    Implements both LLMProvider and StreamingLLMProvider protocols.

    Args:
        client: Anthropic client instance (from anthropic.Anthropic()).
        request_options: Optional default request options to merge with each request.
    """

    def __init__(self, client: Any, *, request_options: Optional[Dict[str, Any]] = None) -> None:
        self.client = client
        self.request_options = request_options or {}

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute an LLM request using Anthropic's API."""
        params = self._build_params(request)
        response = self.client.messages.create(**params)
        return self._parse_response(response)

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """Stream an LLM response chunk by chunk."""
        params = self._build_params(request)
        params["stream"] = True

        response_stream = self.client.messages.create(**params)

        if hasattr(response_stream, "__enter__"):
            with response_stream as stream:
                yield from self._stream_events(stream)
            return

        yield from self._stream_events(response_stream)

    def _build_params(self, request: LLMRequest) -> Dict[str, Any]:
        """Build parameters for Anthropic API call."""
        messages, system = _coerce_messages(request.prompt)
        tools = _claude_tool_specs(request.context.tools)
        tool_choice = request.context.tool_choice

        if tool_choice == "none":
            tools = []
            tool_choice = None

        params: Dict[str, Any] = {
            "model": request.context.model,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = tools
            tool_choice_param = _claude_tool_choice(tool_choice)
            if tool_choice_param:
                params["tool_choice"] = tool_choice_param

        if request.context.temperature is not None:
            params["temperature"] = request.context.temperature

        if request.context.max_tokens is not None:
            params["max_tokens"] = request.context.max_tokens

        request_options = dict(self.request_options)
        extra_options = request.context.metadata.get("request_options")
        if isinstance(extra_options, dict):
            request_options.update(extra_options)
        params.update(request_options)

        return params

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response into LLMResponse."""
        tool_calls: List[ToolCall] = []
        text_parts: List[str] = []

        for block in getattr(response, "content", []) or []:
            block_type = _get_attr(block, "type")
            if block_type == "text":
                text = _get_attr(block, "text")
                if text:
                    text_parts.append(str(text))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=_get_attr(block, "id"),
                        name=_get_attr(block, "name"),
                        arguments=_parse_arguments(_get_attr(block, "input")),
                    )
                )

        usage = _usage_dict(getattr(response, "usage", None))

        return LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            raw=response,
            model=_get_attr(response, "model"),
            finish_reason=_get_attr(response, "stop_reason"),
            usage=usage,
        )

    def _stream_events(self, stream: Iterator[Any]) -> Iterator[LLMStreamChunk]:
        tool_calls_builder: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        pending_usage: Optional[Dict[str, Any]] = None
        finish_emitted = False

        for event in stream:
            event_type = _get_attr(event, "type")

            if event_type == "message_start":
                message = _get_attr(event, "message")
                usage = _usage_dict(_get_attr(message, "usage"))
                if usage:
                    pending_usage = usage
                continue

            if event_type == "content_block_start":
                block = _get_attr(event, "content_block")
                block_type = _get_attr(block, "type")
                if block_type == "tool_use":
                    idx = _get_attr(event, "index") or 0
                    tool_calls_builder[idx] = {
                        "id": _get_attr(block, "id"),
                        "name": _get_attr(block, "name"),
                        "arguments": "",
                    }
                    yield LLMStreamChunk(
                        tool_call_delta={
                            "index": idx,
                            "id": _get_attr(block, "id"),
                            "name": _get_attr(block, "name"),
                            "arguments_delta": "",
                        }
                    )
                continue

            if event_type == "content_block_delta":
                delta = _get_attr(event, "delta")
                delta_type = _get_attr(delta, "type")

                if delta_type == "text_delta":
                    text = _get_attr(delta, "text")
                    if text:
                        yield LLMStreamChunk(text=str(text))
                    continue

                if delta_type == "input_json_delta":
                    idx = _get_attr(event, "index") or 0
                    partial = _get_attr(delta, "partial_json")
                    if partial is None:
                        partial = _get_attr(delta, "text")
                    tool_call = tool_calls_builder.setdefault(
                        idx,
                        {"id": None, "name": None, "arguments": ""},
                    )
                    if partial:
                        tool_call["arguments"] += str(partial)
                    yield LLMStreamChunk(
                        tool_call_delta={
                            "index": idx,
                            "id": tool_call.get("id"),
                            "name": tool_call.get("name"),
                            "arguments_delta": partial,
                        }
                    )
                continue

            if event_type == "message_delta":
                delta = _get_attr(event, "delta")
                stop_reason = _get_attr(delta, "stop_reason")
                if stop_reason:
                    finish_reason = stop_reason
                usage = _usage_dict(_get_attr(delta, "usage"))
                if usage:
                    pending_usage = usage
                continue

            if event_type == "message_stop":
                stop_reason = _get_attr(_get_attr(event, "message"), "stop_reason")
                finish_reason = stop_reason or finish_reason
                yield LLMStreamChunk(finish_reason=finish_reason, usage=pending_usage)
                finish_emitted = True
                continue

        if not finish_emitted and (finish_reason or pending_usage):
            yield LLMStreamChunk(finish_reason=finish_reason, usage=pending_usage)


def _parse_arguments(value: Any) -> Dict[str, Any]:
    """Parse tool call arguments from string or dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return {}


def _claude_tool_specs(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    """Convert ToolSpec list to Claude tool format."""
    specs: List[Dict[str, Any]] = []
    for tool in tools:
        specs.append(
            {
                "name": tool.name,
                "description": tool.description or "Tool call.",
                "input_schema": tool.parameters,
            }
        )
    return specs


def _claude_tool_choice(value: Optional[str]) -> Optional[Dict[str, Any]]:
    """Normalize tool_choice for Claude API."""
    if value is None:
        return None
    if value == "auto":
        return {"type": "auto"}
    return {"type": "tool", "name": value}


def _coerce_messages(prompt: PromptPayload) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Convert PromptPayload to Claude messages format."""
    if prompt.messages is not None:
        system_parts: List[str] = []
        messages: List[Dict[str, Any]] = []
        for message in prompt.messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                if content is not None:
                    system_parts.append(_content_to_text(content))
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            messages.append({"role": role, "content": content})
        return messages, "\n\n".join(part for part in system_parts if part) or None
    if prompt.text is not None:
        return [{"role": "user", "content": prompt.text}], None
    raise ValueError("PromptPayload must include text or messages.")


def _content_to_text(content: Any) -> str:
    """Normalize message content to text for system prompts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        )
    if isinstance(content, dict):
        return str(content.get("text", content))
    return str(content)


def _get_attr(obj: Any, name: str) -> Any:
    """Get attribute from dict-like or object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _usage_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """Convert usage object to dict."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {"value": str(usage)}
