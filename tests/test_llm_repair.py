"""Tests for LLM validation and repair functionality."""

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from coevolved.core.repair import (
    LLMRepairPolicy,
    validated_llm_step,
    default_failure_to_input,
)
from coevolved.core.types import LLMConfig, LLMRequest, LLMResponse, PromptPayload


class User(BaseModel):
    """Test model for validation."""
    name: str
    age: int
    
    @field_validator('age')
    def validate_age(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v


class MockProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, responses):
        """Initialize with list of responses to return."""
        self.responses = responses if isinstance(responses, list) else [responses]
        self.call_count = 0
        self.requests = []
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next mocked response."""
        self.requests.append(request)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


def test_success_first_attempt():
    """Verify that successful validation on first try works without retry."""
    # Valid JSON response
    valid_response = LLMResponse(
        text='{"name": "John", "age": 25}',
        finish_reason="stop"
    )
    provider = MockProvider(valid_response)
    
    step = validated_llm_step(
        prompt_builder=lambda s: "Extract user info",
        provider=provider,
        response_model=User,
        config=LLMConfig(model="gpt-4"),
        repair_policy=LLMRepairPolicy(max_attempts=3),
    )
    
    # Execute step
    state = {"text": "John is 25 years old"}
    result = step(state)
    
    # Assertions
    assert provider.call_count == 1, "Should only call LLM once on success"
    assert "llm_response" in result
    assert isinstance(result["llm_response"], User)
    assert result["llm_response"].name == "John"
    assert result["llm_response"].age == 25


def test_json_in_markdown_code_block():
    """Verify that JSON wrapped in markdown code blocks is extracted."""
    # Response with JSON in markdown code block
    markdown_response = LLMResponse(
        text='```json\n{"name": "Alice", "age": 30}\n```',
        finish_reason="stop"
    )
    provider = MockProvider(markdown_response)
    
    step = validated_llm_step(
        prompt_builder=lambda s: "Extract user",
        provider=provider,
        response_model=User,
        config=LLMConfig(model="gpt-4"),
    )
    
    result = step({})
    
    assert provider.call_count == 1
    assert result["llm_response"].name == "Alice"
    assert result["llm_response"].age == 30


def test_repair_then_success():
    """Verify that failed validation triggers repair and retry."""
    # First response: invalid (negative age)
    invalid_response = LLMResponse(
        text='{"name": "Bob", "age": -5}',
        finish_reason="stop"
    )
    # Second response: valid
    valid_response = LLMResponse(
        text='{"name": "Bob", "age": 5}',
        finish_reason="stop"
    )
    
    provider = MockProvider([invalid_response, valid_response])
    
    step = validated_llm_step(
        prompt_builder=lambda s: "Extract user",
        provider=provider,
        response_model=User,
        config=LLMConfig(model="gpt-4"),
        repair_policy=LLMRepairPolicy(max_attempts=3),
    )
    
    result = step({})
    
    # Should have made 2 attempts
    assert provider.call_count == 2, "Should retry after validation failure"
    assert result["llm_response"].name == "Bob"
    assert result["llm_response"].age == 5


def test_exhaust_attempts():
    """Verify that exhausting max_attempts raises clear error."""
    # Always return invalid JSON
    invalid_response = LLMResponse(
        text='{"name": "Charlie", "age": -10}',
        finish_reason="stop"
    )
    provider = MockProvider(invalid_response)
    
    step = validated_llm_step(
        prompt_builder=lambda s: "Extract user",
        provider=provider,
        response_model=User,
        config=LLMConfig(model="gpt-4"),
        repair_policy=LLMRepairPolicy(max_attempts=3),
    )
    
    # Should raise after exhausting attempts
    with pytest.raises(ValueError) as exc_info:
        step({})
    
    assert provider.call_count == 3, "Should attempt max_attempts times"
    assert "Validation failed after 3 attempts" in str(exc_info.value)


def test_invalid_json_error():
    """Verify that invalid JSON triggers repair."""
    # First response: invalid JSON
    invalid_json = LLMResponse(
        text='not valid json at all',
        finish_reason="stop"
    )
    # Second response: valid
    valid_response = LLMResponse(
        text='{"name": "Diana", "age": 28}',
        finish_reason="stop"
    )
    
    provider = MockProvider([invalid_json, valid_response])
    
    step = validated_llm_step(
        prompt_builder=lambda s: "Extract user",
        provider=provider,
        response_model=User,
        config=LLMConfig(model="gpt-4"),
        repair_policy=LLMRepairPolicy(max_attempts=3),
    )
    
    result = step({})
    
    assert provider.call_count == 2
    assert result["llm_response"].name == "Diana"


def test_custom_failure_to_input():
    """Verify custom repair function is called correctly."""
    calls = []
    
    def custom_repair(failure, state, prompt_payload, response, attempt):
        calls.append({
            "failure": str(failure),
            "attempt": attempt,
        })
        # Return a simple text prompt
        return f"Attempt {attempt + 1}: Please provide valid JSON"
    
    invalid_response = LLMResponse(
        text='{"name": "Eve", "age": -1}',
        finish_reason="stop"
    )
    valid_response = LLMResponse(
        text='{"name": "Eve", "age": 1}',
        finish_reason="stop"
    )
    
    provider = MockProvider([invalid_response, valid_response])
    
    policy = LLMRepairPolicy(
        failure_to_input=custom_repair,
        max_attempts=3,
    )
    
    step = validated_llm_step(
        prompt_builder=lambda s: "Extract user",
        provider=provider,
        response_model=User,
        config=LLMConfig(model="gpt-4"),
        repair_policy=policy,
    )
    
    result = step({})
    
    # Verify custom function was called
    assert len(calls) == 1, "Custom repair should be called once"
    assert calls[0]["attempt"] == 1
    assert "Age must be positive" in calls[0]["failure"]
    assert result["llm_response"].age == 1


def test_default_failure_to_input_with_text_prompt():
    """Test default repair function with text-based prompts."""
    # Create a validation error using Pydantic v2 API
    try:
        User(name="Test", age=-5)  # This will raise ValidationError
    except ValidationError as e:
        failure = e
    
    prompt_payload = PromptPayload(text="Extract user info")
    response = LLMResponse(text='{"name": "Test", "age": -5}')
    
    result = default_failure_to_input(
        failure, {}, prompt_payload, response, 1
    )
    
    assert isinstance(result, PromptPayload)
    assert result.text is not None
    assert "Validation failed on attempt 1" in result.text
    assert "Age must be positive" in result.text
    assert "Extract user info" in result.text


def test_default_failure_to_input_with_messages():
    """Test default repair function with message-based prompts."""
    failure = ValueError("Invalid format")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Extract user info"}
    ]
    prompt_payload = PromptPayload(messages=messages)
    response = LLMResponse(text="invalid json")
    
    result = default_failure_to_input(
        failure, {}, prompt_payload, response, 2
    )
    
    assert isinstance(result, PromptPayload)
    assert result.messages is not None
    assert len(result.messages) == 4  # original 2 + assistant + system
    assert result.messages[-1]["role"] == "system"
    assert "Validation failed on attempt 2" in result.messages[-1]["content"]


def test_backoff_timing():
    """Verify that backoff delay is applied between retries."""
    import time
    
    invalid_response = LLMResponse(text='invalid')
    provider = MockProvider(invalid_response)
    
    policy = LLMRepairPolicy(
        max_attempts=3,
        backoff_seconds=0.1,  # Short delay for testing
        backoff_multiplier=2.0,
    )
    
    step = validated_llm_step(
        prompt_builder=lambda s: "test",
        provider=provider,
        response_model=User,
        config=LLMConfig(model="gpt-4"),
        repair_policy=policy,
    )
    
    start = time.time()
    try:
        step({})
    except ValueError:
        pass  # Expected to fail
    elapsed = time.time() - start
    
    # Should have delays: 0.1s + 0.2s = 0.3s minimum
    # (first delay + second delay, no delay after last attempt)
    assert elapsed >= 0.3, f"Should have backoff delays, got {elapsed}s"


def test_result_key_none():
    """Verify that result_key=None returns parsed model directly."""
    valid_response = LLMResponse(text='{"name": "Frank", "age": 40}')
    provider = MockProvider(valid_response)
    
    step = validated_llm_step(
        prompt_builder=lambda s: "Extract",
        provider=provider,
        response_model=User,
        config=LLMConfig(model="gpt-4"),
        result_key=None,  # Return directly
    )
    
    result = step({})
    
    # Should return the User model directly, not wrapped in state
    assert isinstance(result, User)
    assert result.name == "Frank"
    assert result.age == 40


def test_config_builder():
    """Verify that config_builder is called correctly."""
    config_calls = []
    
    def build_config(state):
        config_calls.append(state)
        return LLMConfig(model="gpt-4", temperature=0.7)
    
    valid_response = LLMResponse(text='{"name": "Grace", "age": 35}')
    provider = MockProvider([valid_response, valid_response])  # Two for potential retry
    
    step = validated_llm_step(
        prompt_builder=lambda s: "Extract",
        provider=provider,
        response_model=User,
        config_builder=build_config,  # Use builder instead of static config
    )
    
    state = {"input": "test"}
    result = step(state)
    
    assert len(config_calls) >= 1, "config_builder should be called"
    assert result["llm_response"].name == "Grace"
