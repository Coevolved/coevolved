"""Tests for prebuilt retry helpers."""

import pytest
from pydantic import BaseModel

from coevolved.base.step import Step
from coevolved.core.types import LLMConfig, LLMRequest, LLMResponse
from coevolved.prebuilt.retry import llm_retry, agent_retry


class TestModel(BaseModel):
    """Test model."""
    value: str


class MockProvider:
    """Mock LLM provider."""
    
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.call_count = 0
    
    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


def test_llm_retry_wrapper():
    """Verify llm_retry wrapper configures repair correctly."""
    valid_response = LLMResponse(text='{"value": "success"}')
    provider = MockProvider(valid_response)
    
    step = llm_retry(
        prompt_builder=lambda s: "test",
        provider=provider,
        response_model=TestModel,
        config=LLMConfig(model="gpt-4"),
        max_attempts=5,
    )
    
    result = step({})
    
    assert provider.call_count == 1
    assert result["llm_response"].value == "success"


def test_llm_retry_with_failure():
    """Verify llm_retry handles validation failure."""
    invalid = LLMResponse(text='invalid json')
    valid = LLMResponse(text='{"value": "fixed"}')
    
    provider = MockProvider([invalid, valid])
    
    step = llm_retry(
        prompt_builder=lambda s: "test",
        provider=provider,
        response_model=TestModel,
        config=LLMConfig(model="gpt-4"),
        max_attempts=3,
    )
    
    result = step({})
    
    assert provider.call_count == 2
    assert result["llm_response"].value == "fixed"


def test_agent_retry_predicate():
    """Verify agent_retry uses custom should_retry predicate."""
    call_count = [0]
    
    def failing_fn(state):
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError("Retry me")
        return {"result": "success"}
    
    base_step = Step(failing_fn, name="test")
    
    def should_retry_fn(exc, state, attempt):
        # Only retry ValueError
        return isinstance(exc, ValueError) and attempt < 3
    
    retry_step = agent_retry(
        base_step,
        should_retry=should_retry_fn,
        max_attempts=5,
        backoff_seconds=0.01,  # Short for testing
    )
    
    result = retry_step({})
    
    assert call_count[0] == 3, "Should retry twice then succeed"
    assert result["result"] == "success"


def test_agent_retry_non_retryable_error():
    """Verify agent_retry doesn't retry when predicate returns False."""
    call_count = [0]
    
    def failing_fn(state):
        call_count[0] += 1
        raise TypeError("Don't retry this")
    
    base_step = Step(failing_fn, name="test")
    
    def should_retry_fn(exc, state, attempt):
        # Only retry ValueError, not TypeError
        return isinstance(exc, ValueError)
    
    retry_step = agent_retry(
        base_step,
        should_retry=should_retry_fn,
        max_attempts=5,
    )
    
    with pytest.raises(TypeError):
        retry_step({})
    
    assert call_count[0] == 1, "Should not retry TypeError"


def test_agent_retry_max_attempts():
    """Verify agent_retry respects max_attempts."""
    call_count = [0]
    
    def always_fail(state):
        call_count[0] += 1
        raise ValueError("Always fails")
    
    base_step = Step(always_fail, name="test")
    
    retry_step = agent_retry(
        base_step,
        should_retry=lambda exc, state, attempt: True,  # Always retry
        max_attempts=4,
        backoff_seconds=0.01,
    )
    
    with pytest.raises(ValueError):
        retry_step({})
    
    assert call_count[0] == 4, "Should attempt max_attempts times"


def test_agent_retry_preserves_step_properties():
    """Verify agent_retry preserves original step properties."""
    class CustomInput(BaseModel):
        x: int
    
    class CustomOutput(BaseModel):
        y: int
    
    base_step = Step(
        lambda s: {"y": s.x * 2},
        name="doubler",
        input_schema=CustomInput,
        output_schema=CustomOutput,
        annotations={"custom": "value"},
    )
    
    retry_step = agent_retry(
        base_step,
        should_retry=lambda exc, state, attempt: False,
        max_attempts=2,
    )
    
    assert retry_step.name == "doubler_agent_retry"
    assert retry_step.input_schema == CustomInput
    assert retry_step.output_schema == CustomOutput
    assert retry_step.annotations["custom"] == "value"
    assert retry_step.annotations["agent_retry_wrapper"] is True
