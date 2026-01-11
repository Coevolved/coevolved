"""Prompt template and rendering utilities.

This module provides a first-class Prompt type for managing prompt templates
with versioning and metadata.
"""

import hashlib
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from coevolved.core.types import PromptPayload


class Prompt(BaseModel):
    """First-class prompt representation with template and metadata.
    
    Prompts use Python string formatting (str.format) for variable substitution.
    
    Attributes:
        id: Optional identifier for prompt versioning.
        version: Optional version string.
        template: Template string with {variable} placeholders.
        metadata: Optional metadata dictionary.
    
    Example:
        >>> prompt = Prompt(
        ...     id="greeting",
        ...     version="1.0",
        ...     template="Hello, {name}! How are you?"
        ... )
        >>> text = prompt.render({"name": "Alice"})  # "Hello, Alice! How are you?"
    """

    id: Optional[str] = None
    version: Optional[str] = None
    template: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def render(self, variables: Dict[str, Any]) -> str:
        """Render the prompt template with given variables.
        
        Args:
            variables: Dictionary of variables for template substitution.
        
        Returns:
            Rendered prompt string.
        
        Raises:
            KeyError: If template references a variable not in variables.
        """
        return self.template.format(**variables)

    def hash(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """Compute a hash of the prompt with variables.
        
        Useful for caching and prompt versioning.
        
        Args:
            variables: Optional variables dictionary.
        
        Returns:
            Hexadecimal hash string.
        """
        payload = {
            "id": self.id,
            "version": self.version,
            "template": self.template,
            "variables": variables or {},
        }
        blob = str(sorted(payload.items()))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_prompt(prompt: Prompt, variables: Dict[str, Any]) -> PromptPayload:
    """Render a prompt and return a PromptPayload with metadata.
    
    Args:
        prompt: Prompt object to render.
        variables: Variables for template substitution.
    
    Returns:
        PromptPayload with rendered text and metadata (id, version, hash).
    """
    text = prompt.render(variables)
    return PromptPayload(
        text=text,
        prompt_id=prompt.id,
        prompt_version=prompt.version,
        prompt_hash=prompt.hash(variables),
    )
