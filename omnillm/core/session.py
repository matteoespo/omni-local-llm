from collections.abc import AsyncIterator, Iterator, Sequence
from threading import Lock
from typing import TYPE_CHECKING, Any

from omnillm.core.types import ChatChunk, ChatMessage, ChatResponse, ToolCall

if TYPE_CHECKING:
    from omnillm.core.manager import LocalLLMManager


class ChatSession:
    """Conversation state that commits a turn only after it completes successfully."""

    def __init__(
        self,
        manager: "LocalLLMManager",
        backend: str,
        model: str,
        system_prompt: str | None = None,
    ):
        self.manager = manager
        self.backend = backend
        self.model = model
        self.messages: list[ChatMessage] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self._state_lock = Lock()
        self._busy = False

    def _claim_turn(self) -> None:
        with self._state_lock:
            if self._busy:
                raise RuntimeError("This session already has an active request.")
            self._busy = True

    def _release_turn(self) -> None:
        with self._state_lock:
            self._busy = False

    def _commit(self, user_message: ChatMessage, content: str, tool_calls: Sequence[ToolCall]) -> None:
        self.messages.append(user_message)
        assistant: ChatMessage = {"role": "assistant", "content": content}
        if tool_calls:
            assistant["tool_calls"] = list(tool_calls)
        self.messages.append(assistant)

    def send(
        self,
        user_input: str,
        *,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatResponse | Iterator[str]:
        self._claim_turn()
        user_message: ChatMessage = {"role": "user", "content": user_input}
        messages = [*self.messages, user_message]
        try:
            result = self.manager.chat(
                self.backend,
                self.model,
                messages,
                stream=stream,
                json_mode=json_mode,
                response_model=response_model,
                response_format=response_format,
                tools=tools,
                **options,
            )
        except Exception:
            self._release_turn()
            raise

        if not stream:
            try:
                if not isinstance(result, ChatResponse):
                    raise RuntimeError("Backend returned a stream for a non-streaming request.")
                self._commit(user_message, result.content, result.tool_calls)
                return result
            finally:
                self._release_turn()

        def wrapped_stream() -> Iterator[str]:
            content: list[str] = []
            tool_calls: list[ToolCall] = []
            try:
                if not isinstance(result, Iterator):
                    raise RuntimeError("Backend returned a non-streaming response for a streaming request.")
                for chunk in result:
                    if not isinstance(chunk, ChatChunk):
                        raise RuntimeError("Backend returned an invalid stream chunk.")
                    content.append(chunk.content)
                    tool_calls.extend(chunk.tool_calls)
                    if chunk.content:
                        yield chunk.content
                self._commit(user_message, "".join(content), tool_calls)
            finally:
                self._release_turn()

        return wrapped_stream()

    async def asend(
        self,
        user_input: str,
        *,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatResponse | AsyncIterator[str]:
        self._claim_turn()
        user_message: ChatMessage = {"role": "user", "content": user_input}
        messages = [*self.messages, user_message]
        try:
            result = await self.manager.achat(
                self.backend,
                self.model,
                messages,
                stream=stream,
                json_mode=json_mode,
                response_model=response_model,
                response_format=response_format,
                tools=tools,
                **options,
            )
        except Exception:
            self._release_turn()
            raise

        if not stream:
            try:
                if not isinstance(result, ChatResponse):
                    raise RuntimeError("Backend returned a stream for a non-streaming request.")
                self._commit(user_message, result.content, result.tool_calls)
                return result
            finally:
                self._release_turn()

        async def wrapped_stream() -> AsyncIterator[str]:
            content: list[str] = []
            tool_calls: list[ToolCall] = []
            try:
                if not isinstance(result, AsyncIterator):
                    raise RuntimeError("Backend returned a non-streaming response for a streaming request.")
                async for chunk in result:
                    if not isinstance(chunk, ChatChunk):
                        raise RuntimeError("Backend returned an invalid stream chunk.")
                    content.append(chunk.content)
                    tool_calls.extend(chunk.tool_calls)
                    if chunk.content:
                        yield chunk.content
                self._commit(user_message, "".join(content), tool_calls)
            finally:
                self._release_turn()

        return wrapped_stream()
