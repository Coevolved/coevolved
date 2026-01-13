# LLM Validation and Retry

Coevolved provides an "Instructor-style" pattern for structured output validation
with automatic repair and retry when parsing or validation fails.

## Quick Start

```python
from pydantic import BaseModel
from coevolved.core import llm_step, LLMConfig
from coevolved.prebuilt import llm_retry

# Define your output schema
class ExtractedUser(BaseModel):
    name: str
    age: int
    email: str

# Create a base LLM step
llm = llm_step(
    prompt_builder=lambda s: f"Extract user info from: {s['text']}",
    provider=my_provider,
    config=LLMConfig(model="gpt-4"),
)

# Wrap with validation and retry
validated_llm = llm_retry(
    llm,
    output_schema=ExtractedUser,
    max_attempts=3,
)

# Use it - validation failures automatically retry with repair context
result = validated_llm({"text": "John Doe, 25 years old, john@example.com"})
user = result["validated_output"]  # ExtractedUser instance
```

## Core Concepts

### LLMRepairPolicy

The `LLMRepairPolicy` configures retry behavior:

```python
from coevolved.core import LLMRepairPolicy

policy = LLMRepairPolicy(
    max_attempts=3,                    # Total attempts including initial
    failure_to_input=my_repair_fn,     # Custom repair function
    retryable_exceptions=(ValueError,), # Which exceptions trigger retry
    backoff_seconds=0.5,               # Initial delay between retries
    backoff_multiplier=2.0,            # Exponential backoff multiplier
)
```

### Custom Repair Functions

Repair functions receive context about the failure and return instructions
for modifying the next attempt:

```python
from coevolved.core import RepairContext, RepairResult

def custom_repair(ctx: RepairContext) -> RepairResult:
    # ctx.error - The exception that occurred
    # ctx.response - The LLM response that failed
    # ctx.attempt - Current attempt number (1-indexed)
    # ctx.state - Current state
    
    # Build a helpful error message
    if "age" in str(ctx.error):
        hint = "Age must be a positive integer, not a string."
    else:
        hint = f"Please fix: {ctx.error}"
    
    return RepairResult(
        # Append messages to the conversation
        messages_append=[{"role": "user", "content": hint}],
        # Or update state
        state_updates={"last_error": str(ctx.error)},
    )

validated_llm = llm_retry(
    llm,
    output_schema=ExtractedUser,
    failure_to_input=custom_repair,
)
```

### Custom Validators

For non-Pydantic validation, provide a custom validator function:

```python
def validate_response(response: LLMResponse) -> dict:
    """Custom validation logic."""
    import json
    
    data = json.loads(response.text)
    
    # Custom validation rules
    if data.get("score", 0) < 0:
        raise ValueError("Score must be non-negative")
    
    if not data.get("summary"):
        raise ValueError("Summary is required")
    
    return data

validated_llm = llm_retry(
    llm,
    validator=validate_response,
    max_attempts=3,
)
```

## Agent Retry

For higher-level agent retry (e.g., retrying an entire agent workflow):

```python
from coevolved.prebuilt import agent_retry, react_agent

# Create an agent
agent = react_agent(planner=planner, tools=tools, max_steps=10)

# Wrap with retry logic
reliable_agent = agent_retry(
    agent,
    max_attempts=3,
    should_retry=lambda exc, state, attempt: isinstance(exc, TimeoutError),
    on_retry=lambda exc, state, attempt: {**state, "retry_count": attempt},
)

result = reliable_agent(initial_state)
```

### Custom Retry Predicates

Control which errors trigger retries:

```python
def should_retry(exc: Exception, state: Any, attempt: int) -> bool:
    # Only retry on specific errors
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, ValidationError) and attempt < 3:
        return True
    return False

reliable_agent = agent_retry(agent, should_retry=should_retry)
```

### State Modification on Retry

Modify state between retry attempts:

```python
def on_retry(exc: Exception, state: Any, attempt: int) -> dict:
    # Reset certain state fields
    return {
        **state,
        "tool_result": None,
        "error_history": state.get("error_history", []) + [str(exc)],
    }

reliable_agent = agent_retry(agent, on_retry=on_retry)
```

## Tracing

Validation attempts are traced via `LLMEvent` with events:
- `validation_success` - When validation passes
- `validation_failure` - When validation fails (before retry)

These events include attempt metadata for debugging retry behavior.

## Error Handling

When all attempts are exhausted, `LLMValidationError` is raised:

```python
from coevolved.core import LLMValidationError

try:
    result = validated_llm(state)
except LLMValidationError as e:
    print(f"Failed after {e.attempts} attempts")
    print(f"Last error: {e.last_error}")
    print(f"Last response: {e.last_response.text}")
```
