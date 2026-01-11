"""Type definitions for LLM interactions.

This module defines Pydantic models and protocols for LLM requests, responses,
prompts, tools, and provider interfaces.
"""

from typing import Any, Dict, Iterator, List, Optional, Protocol

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """Specification for a tool that can be called by an LLM.
    
    Attributes:
        name: Tool name (must match the tool function name).
        description: Optional description for the LLM.
        parameters: JSON schema dictionary for tool arguments.
    """
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class ToolCall(BaseModel):
    """A tool call made by an LLM.
    
    Attributes:
        id: Optional unique identifier for the tool call.
        name: Name of the tool to call.
        arguments: Dictionary of arguments for the tool.
    """
    id: Optional[str] = None
    name: str
    arguments: Dict[str, Any]


class PromptPayload(BaseModel):
    """Payload for an LLM prompt with optional metadata.
    
    Attributes:
        text: Plain text prompt (for simple prompts).
        messages: List of message dictionaries (for chat-style prompts).
        prompt_id: Optional identifier for prompt versioning.
        prompt_version: Optional version string.
        prompt_hash: Optional hash of the prompt content.
    """
    text: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None
    prompt_hash: Optional[str] = None


class LLMConfig(BaseModel):
    """Configuration for an LLM request.
    
    Attributes:
        model: Model identifier (e.g., "gpt-4", "gpt-4o-mini").
        tools: List of available tools for function calling.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum tokens to generate.
        tool_choice: Tool choice mode ("auto", "none", or specific tool name).
        stream: Whether to use streaming response.
        metadata: Additional provider-specific metadata.
    """
    model: str
    tools: List[ToolSpec] = Field(default_factory=list)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tool_choice: Optional[str] = None
    stream: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    """Complete LLM request with prompt and context.
    
    Attributes:
        prompt: Prompt payload (text or messages).
        context: LLM configuration context.
    """
    prompt: PromptPayload
    context: LLMConfig


class LLMResponse(BaseModel):
    """Response from an LLM provider.
    
    Attributes:
        text: Generated text content (if any).
        tool_calls: List of tool calls requested by the LLM.
        raw: Raw response object from the provider (for debugging).
        model: Model identifier used for the response.
        finish_reason: Reason for completion ("stop", "length", "tool_calls", etc.).
        usage: Token usage statistics (if available).
    """
    text: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    raw: Any = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class LLMStreamChunk(BaseModel):
    """A chunk from a streaming LLM response.
    
    Attributes:
        text: Text delta for this chunk (if any).
        tool_call_delta: Partial tool call data (if any).
        finish_reason: Set on final chunk to indicate completion.
        usage: Token usage (typically only on final chunk).
    """
    text: Optional[str] = None
    tool_call_delta: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class LLMProvider(Protocol):
    """Protocol for LLM provider implementations.
    
    Providers must implement the complete method to execute LLM requests.
    Streaming is optional - implement stream() if the provider supports it.
    """
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute an LLM request and return a response.
        
        Args:
            request: LLM request with prompt and context.
        
        Returns:
            LLM response with text, tool calls, and metadata.
        """
        ...


class StreamingLLMProvider(Protocol):
    """Protocol for LLM providers that support streaming.
    
    Extends LLMProvider with a stream() method for token-by-token output.
    
    Example:
        >>> for chunk in provider.stream(request):
        ...     if chunk.text:
        ...         print(chunk.text, end="", flush=True)
        >>> print()  # Final newline
    """
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute an LLM request and return a response."""
        ...
    
    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """Stream an LLM response chunk by chunk.
        
        Args:
            request: LLM request with prompt and context.
        
        Yields:
            LLMStreamChunk for each piece of the response.
            The final chunk will have finish_reason set.
        """
        ...
