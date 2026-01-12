"""LLM and tool step creation utilities.

This module provides helpers for creating Steps that interact with LLM providers
and execute tool functions, with automatic tracing and event emission.
"""

import hashlib
import json
import time
from typing import Any, Callable, Dict, Optional, Union

from pydantic import BaseModel

from coevolved.core.prompt import Prompt, render_prompt
from coevolved.core.events import LLMEvent, PromptEvent, ToolEvent
from coevolved.core.types import (
    LLMConfig,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    PromptPayload,
    ToolCall,
)
from coevolved.base.step import Step
from coevolved.base.tracing import get_current_invocation, get_default_tracer
from coevolved.prebuilt.context import get_agent_context


PromptBuilder = Callable[[Any], Union[str, Prompt, PromptPayload, Dict[str, Any]]]
"""Type alias for prompt builder functions.

A prompt builder takes state and returns a prompt in one of several formats:
- str: Plain text prompt
- Prompt: Prompt object with template
- PromptPayload: Full prompt payload with metadata
- Dict[str, Any]: Dictionary that will be validated as PromptPayload
"""


def llm_step(
    *,
    prompt_builder: PromptBuilder,
    provider: LLMProvider,
    config: Optional[LLMConfig] = None,
    config_builder: Optional[Callable[[Any], LLMConfig]] = None,
    parser: Optional[Callable[[LLMResponse], Any]] = None,
    name: Optional[str] = None,
    annotations: Optional[Dict[str, Any]] = None,
    result_key: Optional[str] = "llm_response",
    log_prompt: bool = False,
) -> Step:
    """Create a Step that renders a prompt, calls an LLM provider, and attaches output.
    
    The step automatically:
    - Renders the prompt from state
    - Emits PromptEvent and LLMEvent for tracing
    - Parses the response (if parser provided)
    - Attaches result to state at result_key (or returns parsed result)
    
    Args:
        prompt_builder: Function that builds a prompt from state. Can return
            str, Prompt, PromptPayload, or dict.
        provider: LLM provider implementation (e.g., OpenAIProvider).
        config: Static LLM configuration (model, tools, temperature, etc.).
        config_builder: Function that builds LLM configuration from state.
            Exactly one of config or config_builder must be provided.
        parser: Optional function to parse LLMResponse into custom format.
        name: Optional step name. Defaults to "llm_step".
        annotations: Optional metadata annotations.
        result_key: Key to attach result in state dict/model. If None, returns
            parsed result directly. Defaults to "llm_response".
        log_prompt: Whether to include full prompt text in PromptEvent.
            Defaults to False for privacy/performance.
    
    Returns:
        Step that executes LLM call and attaches/returns result.
    
    Raises:
        ValueError: If both or neither of config/config_builder are provided.
    
    Example:
        >>> def build_prompt(state):
        ...     return f"Answer: {state['question']}"
        >>> config = LLMConfig(model="gpt-4", temperature=0.7)
        >>> step = llm_step(
        ...     prompt_builder=build_prompt,
        ...     provider=OpenAIProvider(client),
        ...     config=config
        ... )
    """

    if (config is None) == (config_builder is None):
        raise ValueError("Provide exactly one of config or config_builder.")

    def run(state: Any) -> Any:
        pb = prompt_builder(state)
        payload = _coerce_prompt_payload(pb, state)
        if payload.prompt_hash is None:
            payload = payload.model_copy(update={"prompt_hash": _prompt_hash(payload)})

        cfg = config_builder(state) if config_builder else config
        if cfg is None:
            raise ValueError("LLMConfig is required for llm_step execution.")

        tracer = get_default_tracer()
        invocation_ctx = get_current_invocation()
        if invocation_ctx:
            agent_ctx = get_agent_context()
            tracer.emit(
                PromptEvent(
                    run_id=invocation_ctx.run_id,
                    step_id=invocation_ctx.step_id,
                    invocation_id=invocation_ctx.invocation_id,
                    group_hash=invocation_ctx.group_hash,
                    step_name=invocation_ctx.step_name,
                    event=PromptEvent.EVENT_PROMPT,
                    timestamp=time.time(),
                    prompt_id=payload.prompt_id,
                    prompt_version=payload.prompt_version,
                    prompt_hash=payload.prompt_hash,
                    prompt_text=_prompt_text_for_event(payload, log_prompt),
                    annotations=annotations or None,
                    agent_id=agent_ctx.agent_id if agent_ctx else None,
                    agent_type=agent_ctx.agent_type if agent_ctx else None,
                    agent_run_id=agent_ctx.agent_run_id if agent_ctx else None,
                    iteration=agent_ctx.iteration if agent_ctx else None,
                )
            )

        response = provider.complete(LLMRequest(prompt=payload, context=cfg))
        if invocation_ctx:
            tracer.emit(
                LLMEvent(
                    run_id=invocation_ctx.run_id,
                    step_id=invocation_ctx.step_id,
                    invocation_id=invocation_ctx.invocation_id,
                    group_hash=invocation_ctx.group_hash,
                    step_name=invocation_ctx.step_name,
                    event=LLMEvent.EVENT_RESPONSE,
                    timestamp=time.time(),
                    text=response.text,
                    tool_calls=_tool_calls_for_event(response),
                    finish_reason=response.finish_reason,
                    model=response.model,
                    usage=response.usage,
                )
            )
        parsed = parser(response) if parser else response

        if result_key is None:
            return parsed
        if isinstance(state, dict):
            return {**state, result_key: parsed}
        if isinstance(state, BaseModel):
            return state.model_copy(update={result_key: parsed})
        return parsed

    return Step(
        run,
        name=name or "llm_step",
        annotations={**(annotations or {}), "kind": "llm"},
    )


def tool_step(
    fn: Callable[[Any], Any],
    *,
    name: Optional[str] = None,
    input_schema: Optional[type[BaseModel]] = None,
    output_schema: Optional[type[BaseModel]] = None,
    annotations: Optional[Dict[str, Any]] = None,
    result_key: Optional[str] = None,
    tool_schema: Optional[type[BaseModel]] = None,
) -> Step:
    """Wrap a tool function as a Step with validation and tracing.
    
    The step automatically:
    - Validates input/output if schemas provided
    - Emits ToolEvent for tracing
    - Attaches result to state at result_key (or returns result directly)
    
    Args:
        fn: Tool function to wrap. Should accept state and return result.
        name: Optional step name. Defaults to function name.
        input_schema: Optional Pydantic model for input validation.
        output_schema: Optional Pydantic model for output validation.
        annotations: Optional metadata annotations.
        result_key: Key to attach result in state dict/model. If None, returns
            result directly. Useful for tools used in agent loops.
        tool_schema: Optional Pydantic model for tool argument schema.
            Used when generating ToolSpec for LLM tool calling. If not provided,
            uses input_schema.
    
    Returns:
        Step that executes tool function with validation and tracing.
    
    Example:
        >>> class SearchArgs(BaseModel):
        ...     query: str
        >>> def search(state: dict) -> str:
        ...     args = state.get("tool_args", {})
        ...     return f"Results for: {args['query']}"
        >>> tool = tool_step(
        ...     search,
        ...     name="search_tool",
        ...     tool_schema=SearchArgs,
        ...     result_key="tool_result"
        ... )
    """

    step_name = name or getattr(fn, "__name__", "tool_step")

    def run(state: Any) -> Any:
        validated_input = state
        if input_schema:
            validated_input = input_schema.model_validate(state)
        tool_args = _tool_args_for_event(state, validated_input, input_schema)
        tracer = get_default_tracer()
        invocation_ctx = get_current_invocation()
        try:
            output = fn(validated_input)
            if output_schema:
                output = output_schema.model_validate(output)
            if invocation_ctx:
                tracer.emit(
                    ToolEvent(
                        run_id=invocation_ctx.run_id,
                        step_id=invocation_ctx.step_id,
                        invocation_id=invocation_ctx.invocation_id,
                        group_hash=invocation_ctx.group_hash,
                        step_name=invocation_ctx.step_name,
                        event=ToolEvent.EVENT_RESULT,
                        timestamp=time.time(),
                        tool_name=step_name,
                        tool_args=tool_args,
                        tool_result=_normalize_value(output),
                    )
                )
        except Exception as exc:
            if invocation_ctx:
                tracer.emit(
                    ToolEvent(
                        run_id=invocation_ctx.run_id,
                        step_id=invocation_ctx.step_id,
                        invocation_id=invocation_ctx.invocation_id,
                        group_hash=invocation_ctx.group_hash,
                        step_name=invocation_ctx.step_name,
                        event=ToolEvent.EVENT_ERROR,
                        timestamp=time.time(),
                        tool_name=step_name,
                        tool_args=tool_args,
                        error=str(exc),
                    )
                )
            raise
        if result_key is None:
            return output
        if isinstance(state, dict):
            return {**state, result_key: output}
        if isinstance(state, BaseModel):
            return state.model_copy(update={result_key: output})
        return output

    return Step(
        run,
        name=step_name,
        annotations={
            **(annotations or {}),
            "kind": "tool",
            "tool_schema": tool_schema,
        },
    )


def _coerce_prompt_payload(
    value: Union[str, Prompt, PromptPayload, Dict[str, Any]],
    state: Any,
) -> PromptPayload:
    if isinstance(value, PromptPayload):
        return value
    if isinstance(value, Prompt):
        return render_prompt(value, _as_dict(state))
    if isinstance(value, dict):
        return PromptPayload.model_validate(value)
    return PromptPayload(text=str(value))


def _prompt_text_for_event(payload: PromptPayload, log_prompt: bool) -> Optional[str]:
    if not log_prompt:
        return None
    if payload.text is not None:
        return payload.text
    if payload.messages is not None:
        return json.dumps(payload.messages, sort_keys=True, default=_default_json_serializer)
    return None


def _prompt_hash(payload: PromptPayload) -> str:
    if payload.text is not None:
        blob = payload.text
    elif payload.messages is not None:
        blob = json.dumps(payload.messages, sort_keys=True, default=_default_json_serializer)
    else:
        blob = ""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tool_calls_for_event(response: LLMResponse) -> Optional[list[dict]]:
    if not response.tool_calls:
        return None
    calls: list[dict] = []
    for call in response.tool_calls:
        if isinstance(call, ToolCall):
            calls.append(call.model_dump())
        elif isinstance(call, dict):
            calls.append(call)
        else:
            calls.append({"value": str(call)})
    return calls


def _tool_args_for_event(
    state: Any,
    validated_input: Any,
    input_schema: Optional[type[BaseModel]],
) -> Optional[Dict[str, Any]]:
    if input_schema:
        return _coerce_dict(_normalize_value(validated_input))
    if isinstance(state, dict) and "tool_args" in state:
        args = state.get("tool_args")
        return _coerce_dict(_normalize_value(args))
    if isinstance(state, BaseModel) and hasattr(state, "tool_args"):
        return _coerce_dict(_normalize_value(getattr(state, "tool_args")))
    return None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return value


def _coerce_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return {"value": value}


def _default_json_serializer(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _as_dict(state: Any) -> Dict[str, Any]:
    if isinstance(state, BaseModel):
        return state.model_dump()
    if isinstance(state, dict):
        return state
    if hasattr(state, "__dict__"):
        return state.__dict__
    return {"value": state}
