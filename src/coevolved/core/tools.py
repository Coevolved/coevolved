"""Utilities for generating tool specifications from Steps.

This module converts tool Steps into ToolSpec objects that can be passed
to LLM providers for function calling.
"""

from typing import Dict, Iterable, List, Optional

from coevolved.base.step import Step
from coevolved.core.types import ToolSpec


def tool_spec_from_step(step: Step, *, name: Optional[str] = None) -> ToolSpec:
    """Generate a ToolSpec from a tool Step.
    
    Extracts the tool schema (from tool_schema annotation or input_schema)
    and description (from annotations or function docstring) to create a
    ToolSpec suitable for LLM function calling.
    
    Args:
        step: Tool Step created with tool_step().
        name: Optional tool name override. Uses step.name if not provided.
    
    Returns:
        ToolSpec with name, description, and parameter schema.
    
    Raises:
        ValueError: If step doesn't have a Pydantic schema for parameters.
    
    Example:
        >>> tool = tool_step(search_func, tool_schema=SearchArgs)
        >>> spec = tool_spec_from_step(tool)
        >>> # Use spec in LLMConfig.tools
    """
    schema = step.annotations.get("tool_schema") or step.input_schema
    if schema is None or not hasattr(schema, "model_json_schema"):
        raise ValueError(
            f"Tool '{step.name}' must provide a Pydantic tool_schema to build tool specs."
        )
    description = step.annotations.get("description") or _docstring(step)
    return ToolSpec(
        name=name or step.name,
        description=description,
        parameters=schema.model_json_schema(),
    )


def tool_specs_from_steps(steps: Iterable[Step]) -> List[ToolSpec]:
    """Generate ToolSpecs from an iterable of tool Steps.
    
    Args:
        steps: Iterable of tool Steps.
    
    Returns:
        List of ToolSpec objects.
    """
    return [tool_spec_from_step(step) for step in steps]


def tool_specs_from_dict(tools: Dict[str, Step]) -> List[ToolSpec]:
    """Generate ToolSpecs from a dictionary of tool Steps.
    
    Args:
        tools: Dictionary mapping tool names to Steps.
    
    Returns:
        List of ToolSpec objects.
    
    Example:
        >>> tools = {"search": search_tool, "calculate": calc_tool}
        >>> specs = tool_specs_from_dict(tools)
        >>> config = LLMConfig(model="gpt-4", tools=specs)
    """
    return tool_specs_from_steps(tools.values())


def _docstring(step: Step) -> Optional[str]:
    doc = getattr(step.fn, "__doc__", None)
    if not doc:
        return None
    return " ".join(line.strip() for line in doc.splitlines()).strip() or None
