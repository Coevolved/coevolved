"""ReAct-style agent implementation.

This module provides react_agent() for creating agent loops that alternate
between LLM planning and tool execution. Built on top of the agent_loop
primitive for checkpointing and policy support.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel

from coevolved.prebuilt.loop import LoopPolicy, LoopState, agent_loop
from coevolved.base.step import Step
from coevolved.core.types import LLMResponse, ToolCall


def react_agent(
    *,
    planner: Step,
    tools: Dict[str, Step],
    max_steps: int = 5,
    name: Optional[str] = None,
    policy: Optional[LoopPolicy] = None,
    llm_response_key: str = "llm_response",
    tool_result_key: str = "tool_result",
    tool_name_key: str = "tool_name",
    tool_args_key: str = "tool_args",
    final_key: str = "final",
    tool_call_id_key: str = "tool_call_id",
) -> Step:
    """Create a ReAct-style agent loop as a Step.
    
    The agent alternates between:
    1. Calling the planner (LLM step) to get a response
    2. If tool calls are present, executing the first tool
    3. Updating message history and repeating until max_steps or final response
    
    This is built on top of agent_loop, so it supports checkpointing,
    execution policies, and interrupts.
    
    Expected state shape (dict or Pydantic model):
    - messages: list[dict] (optional; used for prompt context and updates)
    - llm_response, tool_name, tool_args, tool_result, final (optional)
    
    Args:
        planner: Step that calls an LLM and attaches LLMResponse at llm_response_key.
            Must be created with llm_step().
        tools: Dictionary mapping tool names to tool Steps. Tools must be created
            with tool_step().
        max_steps: Maximum number of iterations before stopping (default: 5).
            Ignored if policy is provided (use policy.max_iterations instead).
        name: Optional agent name. Defaults to "react_agent".
        policy: Optional LoopPolicy for checkpointing and execution limits.
            If not provided, uses LoopPolicy(max_iterations=max_steps).
        llm_response_key: Key where planner attaches LLMResponse (default: "llm_response").
        tool_result_key: Key where tools attach results (default: "tool_result").
        tool_name_key: Key to store tool name in state (default: "tool_name").
        tool_args_key: Key to store tool arguments in state (default: "tool_args").
        final_key: Key to store final response text (default: "final").
        tool_call_id_key: Key to store tool call ID (default: "tool_call_id").
    
    Returns:
        Step that executes the ReAct agent loop.
    
    Raises:
        TypeError: If tools are not Steps.
        ValueError: If tools are not created with tool_step() or have duplicate names.
        KeyError: If LLM requests a tool that doesn't exist.
        ValueError: If tool doesn't set tool_result_key.
    
    Example:
        >>> planner = llm_step(
        ...     prompt_builder=build_prompt,
        ...     provider=provider,
        ...     context=context
        ... )
        >>> tools = {"search": search_tool, "calculate": calc_tool}
        >>> agent = react_agent(planner=planner, tools=tools, max_steps=10)
        >>> result = agent({"messages": [{"role": "user", "content": "..."}]})
    
    Example with checkpointing:
        >>> from coevolved.base.checkpoint import MemoryCheckpointStore
        >>> store = MemoryCheckpointStore()
        >>> agent = react_agent(
        ...     planner=planner,
        ...     tools=tools,
        ...     policy=LoopPolicy(
        ...         max_iterations=10,
        ...         checkpoint_store=store,
        ...     ),
        ... )
    """
    _validate_tools(tools)
    tool_registry = _build_tool_registry(tools)
    agent_name = name or "react_agent"
    
    # Use provided policy or create one from max_steps
    loop_policy = policy or LoopPolicy(max_iterations=max_steps)
    
    def react_body(state: Any, loop_state: LoopState) -> Any:
        """Execute one iteration of the ReAct loop."""
        # Call planner
        current = planner(state)
        response = _coerce_llm_response(_get(current, llm_response_key))
        
        if response.tool_calls:
            # Execute first tool call
            call = response.tool_calls[0]
            current = _append_tool_call_message(current, call)
            current = _update(
                current,
                {
                    tool_name_key: call.name,
                    tool_args_key: call.arguments,
                    final_key: None,
                    tool_call_id_key: call.id,
                },
            )
            
            tool = _resolve_tool(tools, tool_registry, call.name)
            if tool is None:
                raise KeyError(f"Tool '{call.name}' not found.")
            
            current = tool(current)
            tool_result = _get(current, tool_result_key)
            if tool_result is None:
                raise ValueError(
                    f"Tool '{call.name}' did not set '{tool_result_key}'. "
                    "Use tool_step(..., result_key=...) or return updated state with that key."
                )
            
            current = _append_tool_result_message(
                current,
                tool_result,
                tool_call_id=_get(current, tool_call_id_key),
            )
            return current
        
        # No tool calls - this is a final response
        if response.text:
            current = _append_messages(current, "assistant", response.text)
            current = _update(
                current,
                {
                    final_key: response.text,
                    tool_name_key: None,
                    tool_args_key: None,
                    tool_call_id_key: None,
                },
            )
            return current
        
        # Fallback for empty response
        fallback = str(response.raw or response)
        current = _append_messages(current, "assistant", fallback)
        current = _update(
            current,
            {
                final_key: fallback,
                tool_name_key: None,
                tool_args_key: None,
                tool_call_id_key: None,
            },
        )
        return current
    
    def stop_condition(state: Any) -> bool:
        """Stop when we have a final response."""
        return _get(state, final_key) is not None
    
    return agent_loop(
        body=react_body,
        stop_condition=stop_condition,
        policy=loop_policy,
        name=agent_name,
        agent_type="react",
    )


def _coerce_llm_response(value: Any) -> LLMResponse:
    """Coerce a value to LLMResponse."""
    if isinstance(value, LLMResponse):
        return value
    if isinstance(value, dict):
        return LLMResponse.model_validate(value)
    raise TypeError("planner must return an LLMResponse or dict representing it.")


def _append_tool_call_message(state: Any, call: ToolCall) -> Any:
    """Append a tool call message to the messages list."""
    messages = _get(state, "messages")
    if messages is None:
        return state
    tool_call = {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments,
        },
    }
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call],
    }
    return _update(state, {"messages": list(messages) + [msg]})


def _append_tool_result_message(state: Any, tool_result: Any, tool_call_id: Optional[str]) -> Any:
    """Append a tool result message to the messages list."""
    messages = _get(state, "messages")
    if messages is None:
        return state
    msg = {
        "role": "tool",
        "content": str(tool_result),
    }
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return _update(state, {"messages": list(messages) + [msg]})


def _validate_tools(tools: Dict[str, Step]) -> None:
    """Validate that all tools are properly created Steps."""
    for name, tool in tools.items():
        if not isinstance(tool, Step):
            raise TypeError(f"Tool '{name}' must be a Step created via tool_step.")
        if tool.annotations.get("kind") != "tool":
            raise ValueError(f"Tool '{name}' must be created via tool_step (kind='tool').")
    tool_names = [tool.name for tool in tools.values()]
    if len(set(tool_names)) != len(tool_names):
        raise ValueError("Tool Steps must have unique names.")


def _build_tool_registry(tools: Dict[str, Step]) -> Dict[str, Step]:
    """Build a registry mapping tool names to Steps."""
    return {tool.name: tool for tool in tools.values()}


def _resolve_tool(
    tools: Dict[str, Step],
    tool_registry: Dict[str, Step],
    tool_name: Optional[str],
) -> Optional[Step]:
    """Resolve a tool by name from tools dict or registry."""
    if not tool_name:
        return None
    return tools.get(tool_name) or tool_registry.get(tool_name)


def _get(state: Any, key: str) -> Any:
    """Get a key from state (dict or Pydantic model)."""
    if isinstance(state, dict):
        return state.get(key)
    if isinstance(state, BaseModel):
        return getattr(state, key, None)
    return getattr(state, key, None)


def _update(state: Any, updates: Dict[str, Any]) -> Any:
    """Update state with new values (dict or Pydantic model)."""
    if isinstance(state, dict):
        return {**state, **updates}
    if isinstance(state, BaseModel):
        return state.model_copy(update=updates)
    for k, v in updates.items():
        setattr(state, k, v)
    return state


def _append_messages(state: Any, role: str, content: Any, name: Optional[str] = None) -> Any:
    """Append a message to the messages list in state."""
    messages = _get(state, "messages")
    if messages is None:
        return state
    new_messages = list(messages)
    msg = {"role": role, "content": str(content)}
    if name:
        msg["name"] = name
    new_messages.append(msg)
    return _update(state, {"messages": new_messages})
