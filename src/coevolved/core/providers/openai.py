"""OpenAI provider implementation for LLM requests.

This module provides an OpenAIProvider that implements both LLMProvider and
StreamingLLMProvider protocols for use with OpenAI's API client.
"""

import json
from typing import Any, Dict, Iterator, List, Optional

from coevolved.core.types import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    PromptPayload,
    ToolCall,
    ToolSpec,
)


class OpenAIProvider:
    """LLM provider implementation for OpenAI's API.
    
    Implements both LLMProvider and StreamingLLMProvider protocols.
    
    Args:
        client: OpenAI client instance (from openai.OpenAI()).
        request_options: Optional default request options to merge with each request.
    
    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI(api_key="...")
        >>> provider = OpenAIProvider(client)
        >>> 
        >>> # Non-streaming
        >>> response = provider.complete(request)
        >>> 
        >>> # Streaming
        >>> for chunk in provider.stream(request):
        ...     if chunk.text:
        ...         print(chunk.text, end="", flush=True)
    """
    def __init__(self, client: Any, *, request_options: Optional[Dict[str, Any]] = None) -> None:
        self.client = client
        self.request_options = request_options or {}

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute an LLM request using OpenAI's API.
        
        Args:
            request: LLM request with prompt and context.
        
        Returns:
            LLM response with text, tool calls, and metadata.
        
        Raises:
            Exception: Any exception raised by the OpenAI client.
        """
        params = self._build_params(request)
        response = self.client.chat.completions.create(**params)
        return self._parse_response(response)

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """Stream an LLM response chunk by chunk.
        
        Args:
            request: LLM request with prompt and context.
        
        Yields:
            LLMStreamChunk for each piece of the response.
            The final chunk will have finish_reason set.
        
        Example:
            >>> for chunk in provider.stream(request):
            ...     if chunk.text:
            ...         print(chunk.text, end="", flush=True)
            >>> print()  # Final newline
        """
        params = self._build_params(request)
        params["stream"] = True
        
        # Track tool calls being built across chunks
        tool_calls_builder: Dict[int, Dict[str, Any]] = {}
        
        response_stream = self.client.chat.completions.create(**params)
        
        for chunk in response_stream:
            if not chunk.choices:
                continue
            
            choice = chunk.choices[0]
            delta = choice.delta
            
            # Extract text content
            text = getattr(delta, "content", None)
            
            # Extract tool call deltas
            tool_call_deltas: List[Dict[str, Any]] = []
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_builder:
                        tool_calls_builder[idx] = {
                            "id": tc.id,
                            "name": tc.function.name if tc.function else None,
                            "arguments": "",
                        }
                    if tc.function and tc.function.arguments:
                        tool_calls_builder[idx]["arguments"] += tc.function.arguments
                    
                    # Include delta info in chunk
                    tool_call_deltas.append(
                        {
                            "index": idx,
                            "id": tc.id,
                            "name": tc.function.name if tc.function else None,
                            "arguments_delta": tc.function.arguments if tc.function else None,
                        }
                    )
            
            # Check for finish
            finish_reason = getattr(choice, "finish_reason", None)
            
            # Get usage if present (typically only on final chunk with stream_options)
            usage = None
            if hasattr(chunk, "usage") and chunk.usage:
                usage = _usage_dict(chunk.usage)
            
            if tool_call_deltas:
                for idx, tool_call_delta in enumerate(tool_call_deltas):
                    is_last = idx == len(tool_call_deltas) - 1
                    yield LLMStreamChunk(
                        text=text if idx == 0 else None,
                        tool_call_delta=tool_call_delta,
                        finish_reason=finish_reason if is_last else None,
                        usage=usage if is_last else None,
                    )
                continue

            if text is None and finish_reason is None and usage is None:
                continue

            yield LLMStreamChunk(
                text=text,
                tool_call_delta=None,
                finish_reason=finish_reason,
                usage=usage,
            )

    def _build_params(self, request: LLMRequest) -> Dict[str, Any]:
        """Build parameters for OpenAI API call."""
        messages = _coerce_messages(request.prompt)
        tools = _openai_tool_specs(request.context.tools)
        
        params: Dict[str, Any] = {
            "model": request.context.model,
            "messages": messages,
        }
        
        if tools:
            params["tools"] = tools
            if request.context.tool_choice:
                params["tool_choice"] = _openai_tool_choice(request.context.tool_choice)
        
        if request.context.temperature is not None:
            params["temperature"] = request.context.temperature
        
        if request.context.max_tokens is not None:
            params["max_tokens"] = request.context.max_tokens
        
        # Merge request options
        request_options = dict(self.request_options)
        extra_options = request.context.metadata.get("request_options")
        if isinstance(extra_options, dict):
            request_options.update(extra_options)
        params.update(request_options)
        
        return params

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response into LLMResponse."""
        message = response.choices[0].message
        tool_calls: List[ToolCall] = []
        
        for call in message.tool_calls or []:
            args = _parse_arguments(call.function.arguments)
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=args,
                )
            )
        
        usage = _usage_dict(getattr(response, "usage", None))
        
        return LLMResponse(
            text=message.content,
            tool_calls=tool_calls,
            raw=response,
            model=getattr(response, "model", None),
            finish_reason=response.choices[0].finish_reason,
            usage=usage,
        )


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


def _openai_tool_specs(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
    """Convert ToolSpec list to OpenAI tool format."""
    specs: List[Dict[str, Any]] = []
    for tool in tools:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "Tool call.",
                    "parameters": tool.parameters,
                },
            }
        )
    return specs


def _openai_tool_choice(value: str) -> Any:
    """Normalize tool_choice for OpenAI API."""
    if value in {"auto", "none"}:
        return value
    return {"type": "function", "function": {"name": value}}


def _coerce_messages(prompt: PromptPayload) -> List[Dict[str, Any]]:
    """Convert PromptPayload to OpenAI messages format."""
    if prompt.messages is not None:
        return [_normalize_message(msg) for msg in prompt.messages]
    if prompt.text is not None:
        return [{"role": "user", "content": prompt.text}]
    raise ValueError("PromptPayload must include text or messages.")


def _normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a message dict for OpenAI API."""
    msg = dict(message)
    if "tool_calls" in msg:
        msg["tool_calls"] = [_normalize_tool_call(call) for call in msg["tool_calls"]]
    return msg


def _normalize_tool_call(call: Any) -> Dict[str, Any]:
    """Normalize a tool call for OpenAI API format."""
    if hasattr(call, "model_dump"):
        call = call.model_dump()
    if not isinstance(call, dict):
        call = {"name": str(call)}
    if "function" in call:
        function = dict(call.get("function") or {})
        args = function.get("arguments")
    else:
        function = {
            "name": call.get("name"),
            "arguments": call.get("arguments"),
        }
        args = function.get("arguments")
    if not isinstance(args, str):
        function["arguments"] = json.dumps(args or {})
    call_id = call.get("id")
    return {
        "id": call_id,
        "type": call.get("type") or "function",
        "function": function,
    }


def _usage_dict(usage: Any) -> Optional[Dict[str, Any]]:
    """Convert usage object to dict."""
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {"value": str(usage)}
