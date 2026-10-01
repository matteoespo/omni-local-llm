import pytest

from omnillm import (
    ChatResponse,
    LocalLLMManager,
    ToolRegistry,
    function_to_tool_schema,
    tool,
)
from tests.fakes import RecordingBackend


def add_numbers(a: int, b: int) -> int:
    """Adds two integers together."""
    return a + b


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Gets the current weather for a city."""
    return f"Weather in {city}: 22 degrees {unit}"


def test_function_to_tool_schema():
    schema = function_to_tool_schema(add_numbers)
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "add_numbers"
    assert "Adds two integers" in fn["description"]
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "integer"
    assert params["required"] == ["a", "b"]


def test_tool_decorator():
    schema = getattr(get_weather, "__tool_schema__", None)
    assert schema is not None
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["parameters"]["required"] == ["city"]


def test_tool_registry_execution():
    registry = ToolRegistry([add_numbers, get_weather])
    assert len(registry.schemas) == 2

    # String JSON args
    res = registry.execute("add_numbers", '{"a": 10, "b": 25}')
    assert res == "35"

    # Dict args
    res_weather = registry.execute("get_weather", {"city": "Rome"})
    assert "Rome" in res_weather

    # Unknown tool
    assert "Error: Tool 'unknown' is not registered" in registry.execute("unknown", {})


@pytest.mark.asyncio
async def test_tool_registry_async_execution():
    async def fetch_price(symbol: str) -> float:
        return 142.50

    registry = ToolRegistry([fetch_price])
    res = await registry.aexecute("fetch_price", '{"symbol": "NVDA"}')
    assert res == "142.5"


def test_session_act_loop():
    # Sequence of responses:
    # 1. Assistant requests tool call: add_numbers(a=5, b=15)
    # 2. After receiving tool output, assistant provides final answer
    backend = RecordingBackend()
    responses = [
        ChatResponse(
            content="",
            tool_calls=(
                {
                    "id": "call_1",
                    "function": {"name": "add_numbers", "arguments": '{"a": 5, "b": 15}'},
                },
            ),
        ),
        ChatResponse(content="The sum of 5 and 15 is 20."),
    ]

    # Dynamically return next response on chat()
    def chat_mock(request):
        backend.requests.append(request)
        return responses.pop(0)

    backend.chat = chat_mock

    session = LocalLLMManager({"fake": backend}).create_session("fake", "model")
    final_response = session.act("Calculate 5 + 15", tools=[add_numbers])

    assert final_response.content == "The sum of 5 and 15 is 20."
    assert len(session.messages) == 4
    # User message
    assert session.messages[0]["role"] == "user"
    # Assistant tool call
    assert session.messages[1]["role"] == "assistant"
    assert session.messages[1]["tool_calls"][0]["function"]["name"] == "add_numbers"
    # Tool output
    assert session.messages[2]["role"] == "tool"
    assert session.messages[2]["content"] == "20"
    # Assistant final answer
    assert session.messages[3]["role"] == "assistant"
    assert session.messages[3]["content"] == "The sum of 5 and 15 is 20."


@pytest.mark.asyncio
async def test_session_aact_loop():
    backend = RecordingBackend()
    responses = [
        ChatResponse(
            content="",
            tool_calls=(
                {
                    "id": "call_10",
                    "function": {"name": "get_weather", "arguments": '{"city": "Milan"}'},
                },
            ),
        ),
        ChatResponse(content="The weather in Milan is pleasant."),
    ]

    async def achat_mock(request):
        backend.requests.append(request)
        return responses.pop(0)

    backend.achat = achat_mock

    session = LocalLLMManager({"fake": backend}).create_session("fake", "model")
    final_response = await session.aact("How is Milan?", tools=[get_weather])

    assert final_response.content == "The weather in Milan is pleasant."
    assert len(session.messages) == 4
    assert session.messages[2]["role"] == "tool"
    assert "Milan" in session.messages[2]["content"]
