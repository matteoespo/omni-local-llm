import pytest

from omnillm import ChatChunk, ChatResponse, LocalLLMManager
from tests.fakes import RecordingBackend


def test_session_commits_a_completed_response_to_history():
    backend = RecordingBackend(response=ChatResponse(content="I am an AI."))
    session = LocalLLMManager({"fake": backend}).create_session("fake", "test-model", system_prompt="Sys")

    response = session.send("Who are you?")

    assert response.content == "I am an AI."
    assert session.messages == [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am an AI."},
    ]


def test_session_commits_streamed_content_and_tool_calls_after_consumption():
    backend = RecordingBackend(
        chunks=(
            ChatChunk(content="Use "),
            ChatChunk(tool_calls=({"id": "call_1", "function": {"name": "weather"}},), finish_reason="tool_calls"),
        )
    )
    session = LocalLLMManager({"fake": backend}).create_session("fake", "test-model")

    assert list(session.send("Weather?", stream=True)) == ["Use "]
    assert session.messages == [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "Use ",
            "tool_calls": [{"id": "call_1", "function": {"name": "weather"}}],
        },
    ]


def test_session_does_not_commit_a_failed_turn_and_can_be_reused():
    backend = RecordingBackend(error=RuntimeError("backend failed"))
    session = LocalLLMManager({"fake": backend}).create_session("fake", "test-model")

    with pytest.raises(RuntimeError, match="backend failed"):
        session.send("Will this persist?")

    assert session.messages == []
    backend.error = None
    backend.response = ChatResponse(content="Recovered")
    assert session.send("Try again").content == "Recovered"
    assert len(session.messages) == 2


@pytest.mark.asyncio
async def test_async_session_commits_streamed_content():
    backend = RecordingBackend(chunks=(ChatChunk(content="Async "), ChatChunk(content="stream")))
    session = LocalLLMManager({"fake": backend}).create_session("fake", "test-model")

    stream = await session.asend("Hello", stream=True)

    assert [chunk async for chunk in stream] == ["Async ", "stream"]
    assert session.messages[-1] == {"role": "assistant", "content": "Async stream"}


def test_session_with_response_model():
    from pydantic import BaseModel

    class Item(BaseModel):
        count: int

    backend = RecordingBackend(response=ChatResponse(content='{"count": 42}'))
    session = LocalLLMManager({"fake": backend}).create_session("fake", "test-model")

    response = session.send("How many?", response_model=Item)

    assert isinstance(response, ChatResponse)
    assert response.parsed == Item(count=42)
    assert session.messages[-1] == {"role": "assistant", "content": '{"count": 42}'}


def test_session_validation_max_turns_and_budget():
    backend = RecordingBackend(response=ChatResponse(content="ok"))
    manager = LocalLLMManager({"fake": backend})

    with pytest.raises(ValueError, match="max_turns must be at least 1"):
        manager.create_session("fake", "model", max_turns=0)

    with pytest.raises(ValueError, match="max_tokens_budget must be at least 1"):
        manager.create_session("fake", "model", max_tokens_budget=-5)


def test_session_sliding_window_max_turns_preserves_system_prompt():
    backend = RecordingBackend(response=ChatResponse(content="assistant response"))
    session = LocalLLMManager({"fake": backend}).create_session(
        "fake",
        "model",
        system_prompt="System prompt",
        max_turns=2,
    )

    session.send("Turn 1")
    session.send("Turn 2")
    assert session.turn_count == 2
    assert len(session.messages) == 5  # System + 2 turns (4 messages)

    session.send("Turn 3")
    assert session.turn_count == 2
    assert session.messages[0] == {"role": "system", "content": "System prompt"}
    # Oldest turn (Turn 1) should be evicted
    assert [m["content"] for m in session.messages if m.get("role") == "user"] == ["Turn 2", "Turn 3"]


def test_session_token_budget_pruning():
    backend = RecordingBackend(response=ChatResponse(content="short reply"))
    # Estimate: ~4 tokens per message + len // 4.
    # A 50-char string is ~16 tokens. A turn is ~32 tokens.
    session = LocalLLMManager({"fake": backend}).create_session(
        "fake",
        "model",
        system_prompt="Sys",
        max_tokens_budget=40,
    )

    session.send("Message 1: " + "a" * 40)
    session.send("Message 2: " + "b" * 40)

    # Budget of 40 should not fit both long turns + system prompt, so oldest turn is pruned
    assert session.turn_count == 1
    assert session.messages[0] == {"role": "system", "content": "Sys"}
    assert session.messages[1]["content"] == "Message 2: " + "b" * 40


def test_session_proactive_pruning_before_sending():
    backend = RecordingBackend(response=ChatResponse(content="reply"))
    session = LocalLLMManager({"fake": backend}).create_session(
        "fake",
        "model",
        system_prompt="Sys",
        max_tokens_budget=50,
    )

    session.send("Turn 1")
    # Now send an incoming message that is so large it forces pruning of Turn 1 *before* dispatch
    large_turn_2 = "Turn 2: " + "x" * 200
    session.send(large_turn_2)

    # Inspect the messages recorded by the backend adapter during the 2nd request
    recorded_messages = backend.requests[-1].messages
    # Turn 1 must have been evicted proactively before manager.chat was called
    user_prompts = [m["content"] for m in recorded_messages if m.get("role") == "user"]
    assert user_prompts == [large_turn_2]


def test_session_clear_and_reset():
    backend = RecordingBackend(response=ChatResponse(content="reply"))
    session = LocalLLMManager({"fake": backend}).create_session(
        "fake",
        "model",
        system_prompt="Initial system prompt",
    )

    session.send("Hello")
    assert session.turn_count == 1
    assert session.estimated_tokens > 0

    # clear() preserves the initial system prompt
    session.clear()
    assert session.turn_count == 0
    assert session.messages == [{"role": "system", "content": "Initial system prompt"}]
    assert session.system_prompt == "Initial system prompt"

    # reset() without args clears system prompt too
    session.reset()
    assert session.turn_count == 0
    assert session.messages == []
    assert session.system_prompt is None

    # reset() with new prompt
    session.reset("New prompt")
    assert session.messages == [{"role": "system", "content": "New prompt"}]
    assert session.system_prompt == "New prompt"


def test_session_busy_state_prevents_concurrent_operations():
    backend = RecordingBackend(response=ChatResponse(content="ok"))
    session = LocalLLMManager({"fake": backend}).create_session("fake", "model")

    session._claim_turn()
    with pytest.raises(RuntimeError, match="already has an active request"):
        session._claim_turn()

    with pytest.raises(RuntimeError, match="Cannot clear session while a request is in progress"):
        session.clear()

    with pytest.raises(RuntimeError, match="Cannot reset session while a request is in progress"):
        session.reset()

    session._release_turn()
