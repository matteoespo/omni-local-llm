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
