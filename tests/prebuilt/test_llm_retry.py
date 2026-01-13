"""Tests for prebuilt LLM and agent retry utilities."""

import pytest
from pydantic import BaseModel

from coevolved.base.step import Step
from coevolved.core.llm_repair import LLMValidationError, RepairContext, RepairResult
from coevolved.core.types import LLMResponse
from coevolved.prebuilt.llm_retry import agent_retry, llm_retry


class ExtractedData(BaseModel):
    value: str
    count: int


def make_llm_step(responses: list, name: str = "test_llm"):
    """Create a mock LLM step that returns responses in sequence."""
    call_count = [0]
    
    def run(state):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        response = responses[idx]
        if isinstance(state, dict):
            return {**state, "llm_response": response}
        return response
    
    return Step(run, name=name, annotations={"kind": "llm"})


class TestLLMRetry:
    def test_llm_retry_requires_llm_step(self):
        regular_step = Step(lambda s: s, name="not_llm")
        
        with pytest.raises(ValueError, match="requires an LLM step"):
            llm_retry(regular_step, output_schema=ExtractedData)

    def test_success_on_first_attempt(self):
        response = LLMResponse(text='{"value": "test", "count": 42}')
        step = make_llm_step([response])
        
        wrapped = llm_retry(step, output_schema=ExtractedData)
        result = wrapped({})
        
        assert "validated_output" in result
        assert result["validated_output"].value == "test"
        assert result["validated_output"].count == 42

    def test_success_after_repair(self):
        bad_response = LLMResponse(text='{"value": "test", "count": "not a number"}')
        good_response = LLMResponse(text='{"value": "test", "count": 42}')
        step = make_llm_step([bad_response, good_response])
        
        wrapped = llm_retry(step, output_schema=ExtractedData, max_attempts=3)
        result = wrapped({"messages": []})
        
        assert result["validated_output"].count == 42

    def test_exhausts_attempts_and_raises(self):
        bad_response = LLMResponse(text='{"value": "test", "count": "invalid"}')
        step = make_llm_step([bad_response, bad_response, bad_response])
        
        wrapped = llm_retry(step, output_schema=ExtractedData, max_attempts=3)
        
        with pytest.raises(LLMValidationError) as exc_info:
            wrapped({})
        
        assert exc_info.value.attempts == 3

    def test_custom_validator(self):
        response = LLMResponse(text="custom format: test=42")
        step = make_llm_step([response])
        
        def custom_validator(resp: LLMResponse) -> dict:
            parts = resp.text.replace("custom format: ", "").split("=")
            return {"key": parts[0], "val": int(parts[1])}
        
        wrapped = llm_retry(step, validator=custom_validator)
        result = wrapped({})
        
        assert result["validated_output"] == {"key": "test", "val": 42}

    def test_custom_repair_function(self):
        repair_calls = []
        
        def my_repair(ctx: RepairContext) -> RepairResult:
            repair_calls.append(ctx.attempt)
            return RepairResult(
                state_updates={"hint": f"retry {ctx.attempt}"}
            )
        
        bad_response = LLMResponse(text="bad")
        good_response = LLMResponse(text='{"value": "ok", "count": 1}')
        step = make_llm_step([bad_response, good_response])
        
        wrapped = llm_retry(
            step,
            output_schema=ExtractedData,
            failure_to_input=my_repair,
        )
        result = wrapped({})
        
        assert len(repair_calls) == 1
        assert result["validated_output"].value == "ok"

    def test_result_key_none_returns_directly(self):
        response = LLMResponse(text='{"value": "direct", "count": 99}')
        step = make_llm_step([response])
        
        wrapped = llm_retry(step, output_schema=ExtractedData, result_key=None)
        result = wrapped({})
        
        assert isinstance(result, ExtractedData)
        assert result.value == "direct"

    def test_handles_json_with_markdown_fences(self):
        response = LLMResponse(text='```json\n{"value": "fenced", "count": 5}\n```')
        step = make_llm_step([response])
        
        wrapped = llm_retry(step, output_schema=ExtractedData)
        result = wrapped({})
        
        assert result["validated_output"].value == "fenced"

    def test_step_annotations_preserved(self):
        step = make_llm_step([LLMResponse(text="{}")])
        step.annotations["custom"] = "value"
        
        wrapped = llm_retry(step, max_attempts=5)
        
        assert wrapped.annotations["custom"] == "value"
        assert wrapped.annotations["retry_wrapper"] is True
        assert wrapped.annotations["max_attempts"] == 5


class TestAgentRetry:
    def test_success_on_first_attempt(self):
        step = Step(lambda s: {**s, "result": "done"}, name="agent")
        
        wrapped = agent_retry(step, max_attempts=3)
        result = wrapped({"input": "test"})
        
        assert result["result"] == "done"

    def test_retries_on_failure_then_succeeds(self):
        call_count = [0]
        
        def run(state):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("temporary failure")
            return {**state, "result": "success"}
        
        step = Step(run, name="flaky_agent")
        wrapped = agent_retry(step, max_attempts=3)
        result = wrapped({})
        
        assert call_count[0] == 3
        assert result["result"] == "success"

    def test_exhausts_attempts_and_raises(self):
        def run(state):
            raise ValueError("always fails")
        
        step = Step(run, name="failing_agent")
        wrapped = agent_retry(step, max_attempts=3)
        
        with pytest.raises(ValueError, match="always fails"):
            wrapped({})

    def test_custom_should_retry_predicate(self):
        call_count = [0]
        
        def run(state):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("timed out")
            raise ValueError("different error")
        
        def only_retry_timeout(exc, state, attempt):
            return isinstance(exc, TimeoutError)
        
        step = Step(run, name="agent")
        wrapped = agent_retry(step, max_attempts=3, should_retry=only_retry_timeout)
        
        with pytest.raises(ValueError, match="different error"):
            wrapped({})
        
        assert call_count[0] == 2

    def test_on_retry_callback_modifies_state(self):
        call_count = [0]
        states_seen = []
        
        def run(state):
            call_count[0] += 1
            states_seen.append(state.copy())
            if call_count[0] < 3:
                raise ValueError("retry")
            return state
        
        def on_retry(exc, state, attempt):
            return {**state, "retry_count": attempt}
        
        step = Step(run, name="agent")
        wrapped = agent_retry(step, max_attempts=3, on_retry=on_retry)
        result = wrapped({"initial": True})
        
        assert states_seen[0] == {"initial": True}
        assert states_seen[1] == {"initial": True, "retry_count": 1}
        assert states_seen[2] == {"initial": True, "retry_count": 2}

    def test_on_retry_callback_returning_none_keeps_state(self):
        call_count = [0]
        callback_called = [False]
        
        def run(state):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry")
            return state
        
        def on_retry(exc, state, attempt):
            callback_called[0] = True
            return None
        
        step = Step(run, name="agent")
        wrapped = agent_retry(step, max_attempts=3, on_retry=on_retry)
        result = wrapped({"value": 1})
        
        assert callback_called[0] is True
        assert result == {"value": 1}

    def test_step_annotations_include_retry_info(self):
        step = Step(lambda s: s, name="agent", annotations={"kind": "agent"})
        
        wrapped = agent_retry(step, max_attempts=5)
        
        assert wrapped.annotations["kind"] == "agent"
        assert wrapped.annotations["agent_retry_wrapper"] is True
        assert wrapped.annotations["max_attempts"] == 5
        assert wrapped.name == "agent_retry"
