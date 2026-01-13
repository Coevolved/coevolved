"""Tests for LLM validation and repair infrastructure."""

import pytest
from pydantic import BaseModel, ValidationError

from coevolved.core.llm_repair import (
    DEFAULT_RETRYABLE_EXCEPTIONS,
    LLMRepairPolicy,
    LLMValidationError,
    RepairContext,
    RepairResult,
    apply_repair_result,
    default_validation_repair,
    validated_llm_call,
)
from coevolved.core.types import LLMResponse, PromptPayload


class ExtractedUser(BaseModel):
    name: str
    age: int


class TestRepairContext:
    def test_repair_context_holds_all_fields(self):
        error = ValueError("test error")
        ctx = RepairContext(
            error=error,
            state={"key": "value"},
            prompt_payload=PromptPayload(text="test"),
            response=LLMResponse(text="bad response"),
            attempt=1,
            max_attempts=3,
        )
        assert ctx.error is error
        assert ctx.state == {"key": "value"}
        assert ctx.prompt_payload.text == "test"
        assert ctx.response.text == "bad response"
        assert ctx.attempt == 1
        assert ctx.max_attempts == 3


class TestRepairResult:
    def test_apply_repair_result_with_state_updates_dict(self):
        state = {"a": 1, "b": 2}
        payload = PromptPayload(text="test")
        repair = RepairResult(state_updates={"b": 3, "c": 4})
        
        new_state, new_payload = apply_repair_result(state, payload, repair)
        
        assert new_state == {"a": 1, "b": 3, "c": 4}
        assert new_payload is payload

    def test_apply_repair_result_with_state_updates_pydantic(self):
        class State(BaseModel):
            a: int
            b: int = 0
        
        state = State(a=1, b=2)
        payload = PromptPayload(text="test")
        repair = RepairResult(state_updates={"b": 5})
        
        new_state, new_payload = apply_repair_result(state, payload, repair)
        
        assert new_state.a == 1
        assert new_state.b == 5

    def test_apply_repair_result_with_messages_append(self):
        state = {}
        payload = PromptPayload(messages=[{"role": "user", "content": "hello"}])
        repair = RepairResult(
            messages_append=[{"role": "user", "content": "fix this"}]
        )
        
        new_state, new_payload = apply_repair_result(state, payload, repair)
        
        assert len(new_payload.messages) == 2
        assert new_payload.messages[1]["content"] == "fix this"

    def test_apply_repair_result_converts_text_to_messages(self):
        state = {}
        payload = PromptPayload(text="initial prompt")
        repair = RepairResult(
            messages_append=[{"role": "user", "content": "repair context"}]
        )
        
        new_state, new_payload = apply_repair_result(state, payload, repair)
        
        assert new_payload.messages is not None
        assert len(new_payload.messages) == 2
        assert new_payload.messages[0]["content"] == "initial prompt"
        assert new_payload.messages[1]["content"] == "repair context"

    def test_apply_repair_result_with_text_append(self):
        state = {}
        payload = PromptPayload(text="original")
        repair = RepairResult(prompt_text_append="additional context")
        
        new_state, new_payload = apply_repair_result(state, payload, repair)
        
        assert "original" in new_payload.text
        assert "additional context" in new_payload.text


class TestDefaultValidationRepair:
    def test_default_repair_formats_validation_error(self):
        try:
            ExtractedUser(name="John", age="not a number")
        except ValidationError as e:
            ctx = RepairContext(
                error=e,
                state={},
                prompt_payload=PromptPayload(text="test"),
                response=LLMResponse(text='{"name": "John", "age": "not a number"}'),
                attempt=1,
                max_attempts=3,
            )
            result = default_validation_repair(ctx)
            
            assert result.messages_append is not None
            assert len(result.messages_append) == 1
            msg = result.messages_append[0]["content"]
            assert "validation" in msg.lower() or "error" in msg.lower()
            assert "age" in msg.lower()

    def test_default_repair_formats_generic_error(self):
        ctx = RepairContext(
            error=ValueError("JSON parse failed"),
            state={},
            prompt_payload=PromptPayload(text="test"),
            response=LLMResponse(text="not valid json"),
            attempt=1,
            max_attempts=3,
        )
        result = default_validation_repair(ctx)
        
        assert result.messages_append is not None
        msg = result.messages_append[0]["content"]
        assert "JSON parse failed" in msg
        assert "not valid json" in msg


class TestLLMRepairPolicy:
    def test_policy_defaults(self):
        policy = LLMRepairPolicy()
        assert policy.max_attempts == 3
        assert policy.backoff_seconds == 0.0
        assert policy.backoff_multiplier == 2.0

    def test_is_retryable_with_validation_error(self):
        policy = LLMRepairPolicy()
        try:
            ExtractedUser(name="test", age="bad")
        except ValidationError as e:
            assert policy.is_retryable(e) is True

    def test_is_retryable_with_custom_exceptions(self):
        policy = LLMRepairPolicy(retryable_exceptions=(KeyError,))
        assert policy.is_retryable(KeyError("test")) is True
        assert policy.is_retryable(ValueError("test")) is False


class TestValidatedLLMCall:
    def test_success_on_first_attempt(self):
        call_count = 0
        
        def llm_fn(state, payload):
            nonlocal call_count
            call_count += 1
            return LLMResponse(text='{"name": "John", "age": 25}')
        
        def validator(response):
            import json
            data = json.loads(response.text)
            return ExtractedUser.model_validate(data)
        
        result = validated_llm_call(
            llm_fn=llm_fn,
            prompt_payload=PromptPayload(text="extract user"),
            state={},
            validator=validator,
            policy=LLMRepairPolicy(max_attempts=3),
        )
        
        assert call_count == 1
        assert result.name == "John"
        assert result.age == 25

    def test_success_after_repair(self):
        call_count = 0
        
        def llm_fn(state, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return LLMResponse(text='{"name": "John", "age": "twenty"}')
            return LLMResponse(text='{"name": "John", "age": 25}')
        
        def validator(response):
            import json
            data = json.loads(response.text)
            return ExtractedUser.model_validate(data)
        
        result = validated_llm_call(
            llm_fn=llm_fn,
            prompt_payload=PromptPayload(text="extract user"),
            state={},
            validator=validator,
            policy=LLMRepairPolicy(max_attempts=3),
        )
        
        assert call_count == 2
        assert result.name == "John"
        assert result.age == 25

    def test_exhausts_attempts_and_raises(self):
        call_count = 0
        
        def llm_fn(state, payload):
            nonlocal call_count
            call_count += 1
            return LLMResponse(text='{"name": "John", "age": "invalid"}')
        
        def validator(response):
            import json
            data = json.loads(response.text)
            return ExtractedUser.model_validate(data)
        
        with pytest.raises(LLMValidationError) as exc_info:
            validated_llm_call(
                llm_fn=llm_fn,
                prompt_payload=PromptPayload(text="extract user"),
                state={},
                validator=validator,
                policy=LLMRepairPolicy(max_attempts=3),
            )
        
        assert call_count == 3
        assert exc_info.value.attempts == 3
        assert exc_info.value.last_response is not None
        assert isinstance(exc_info.value.last_error, ValidationError)

    def test_custom_repair_function_is_called(self):
        repair_calls = []
        
        def custom_repair(ctx: RepairContext) -> RepairResult:
            repair_calls.append(ctx.attempt)
            return RepairResult(
                messages_append=[{"role": "user", "content": f"Attempt {ctx.attempt} failed"}]
            )
        
        call_count = 0
        payloads_received = []
        
        def llm_fn(state, payload):
            nonlocal call_count
            call_count += 1
            payloads_received.append(payload)
            if call_count < 3:
                return LLMResponse(text="bad")
            return LLMResponse(text='{"name": "Jane", "age": 30}')
        
        def validator(response):
            import json
            data = json.loads(response.text)
            return ExtractedUser.model_validate(data)
        
        policy = LLMRepairPolicy(max_attempts=3, failure_to_input=custom_repair)
        
        result = validated_llm_call(
            llm_fn=llm_fn,
            prompt_payload=PromptPayload(messages=[{"role": "user", "content": "extract"}]),
            state={},
            validator=validator,
            policy=policy,
        )
        
        assert repair_calls == [1, 2]
        assert result.name == "Jane"
        assert result.age == 30
        assert len(payloads_received[2].messages) > len(payloads_received[0].messages)

    def test_non_retryable_exception_raises_immediately(self):
        call_count = 0
        
        def llm_fn(state, payload):
            nonlocal call_count
            call_count += 1
            return LLMResponse(text="test")
        
        def validator(response):
            raise RuntimeError("Not retryable")
        
        policy = LLMRepairPolicy(
            max_attempts=3,
            retryable_exceptions=(ValidationError,),
        )
        
        with pytest.raises(RuntimeError, match="Not retryable"):
            validated_llm_call(
                llm_fn=llm_fn,
                prompt_payload=PromptPayload(text="test"),
                state={},
                validator=validator,
                policy=policy,
            )
        
        assert call_count == 1


class TestLLMValidationError:
    def test_error_contains_all_context(self):
        original = ValueError("parse failed")
        response = LLMResponse(text="bad data")
        
        error = LLMValidationError(
            "Validation failed",
            last_error=original,
            attempts=3,
            last_response=response,
        )
        
        assert error.last_error is original
        assert error.attempts == 3
        assert error.last_response is response
        assert "Validation failed" in str(error)
