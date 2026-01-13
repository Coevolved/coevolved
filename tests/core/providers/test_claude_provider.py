from coevolved.core.providers import ClaudeProvider
from coevolved.core.types import LLMConfig, LLMRequest, PromptPayload, ToolCall, ToolSpec


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeToolUseBlock:
    def __init__(self, tool_id: str, name: str, input_data) -> None:
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = input_data


class FakeResponse:
    def __init__(self, content, *, model: str, stop_reason: str, usage) -> None:
        self.content = content
        self.model = model
        self.stop_reason = stop_reason
        self.usage = usage


class FakeStream:
    def __init__(self, events) -> None:
        self.events = events

    def __iter__(self):
        return iter(self.events)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeMessages:
    def __init__(self, response=None, stream=None) -> None:
        self.response = response
        self.stream = stream
        self.last_params = None

    def create(self, **params):
        self.last_params = params
        if params.get("stream"):
            return self.stream
        return self.response


class FakeClient:
    def __init__(self, response=None, stream=None) -> None:
        self.messages = FakeMessages(response=response, stream=stream)


def test_claude_complete_parses_text_and_tool_calls():
    response = FakeResponse(
        content=[
            FakeTextBlock("Hello "),
            FakeToolUseBlock("tool-1", "search", {"query": "docs"}),
            FakeTextBlock("done"),
        ],
        model="claude-test",
        stop_reason="tool_use",
        usage={"input_tokens": 3, "output_tokens": 2},
    )
    client = FakeClient(response=response)
    provider = ClaudeProvider(client)
    tool_spec = ToolSpec(
        name="search",
        description="Search docs",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    request = LLMRequest(
        prompt=PromptPayload(text="Hi"),
        context=LLMConfig(
            model="claude-test",
            max_tokens=10,
            tools=[tool_spec],
            tool_choice="auto",
        ),
    )

    result = provider.complete(request)

    assert result.text == "Hello done"
    assert result.tool_calls == [
        ToolCall(id="tool-1", name="search", arguments={"query": "docs"})
    ]
    assert result.finish_reason == "tool_use"
    assert result.usage == {"input_tokens": 3, "output_tokens": 2}

    params = client.messages.last_params
    assert params["messages"] == [{"role": "user", "content": "Hi"}]
    assert params["tools"][0]["input_schema"] == tool_spec.parameters
    assert params["tool_choice"] == {"type": "auto"}


def test_claude_stream_emits_chunks_and_finish():
    events = [
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello "},
        },
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "tool-1",
                "name": "search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{\"query\": \"docs\"}"},
        },
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        },
        {"type": "message_stop", "message": {"stop_reason": "tool_use"}},
    ]
    client = FakeClient(stream=FakeStream(events))
    provider = ClaudeProvider(client)
    request = LLMRequest(
        prompt=PromptPayload(text="Hi"),
        context=LLMConfig(model="claude-test", max_tokens=5),
    )

    chunks = list(provider.stream(request))

    assert chunks[0].text == "Hello "
    assert any(
        chunk.tool_call_delta
        and chunk.tool_call_delta.get("name") == "search"
        for chunk in chunks
    )
    assert chunks[-1].finish_reason == "tool_use"
    assert chunks[-1].usage == {"input_tokens": 2, "output_tokens": 1}
