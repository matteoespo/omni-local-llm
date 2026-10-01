from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, overload

from omnillm.core.types import ChatChunk, ChatMessage, ChatResponse, ToolCall

if TYPE_CHECKING:
    from omnillm.core.manager import LocalLLMManager


class ChatSession:
    """Conversation state that commits a turn only after it completes successfully,

    with support for context window pruning and token budget management.
    """

    def __init__(
        self,
        manager: "LocalLLMManager",
        backend: str,
        model: str,
        system_prompt: str | None = None,
        *,
        max_turns: int | None = None,
        max_tokens_budget: int | None = None,
        strategy: Literal["full", "sliding_window"] = "full",
    ):
        if max_turns is not None and max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        if max_tokens_budget is not None and max_tokens_budget < 1:
            raise ValueError("max_tokens_budget must be at least 1.")

        self.manager = manager
        self.backend = backend
        self.model = model
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_tokens_budget = max_tokens_budget
        self.strategy: Literal["full", "sliding_window"] = (
            "sliding_window"
            if (max_turns is not None or max_tokens_budget is not None) and strategy == "full"
            else strategy
        )
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

    @staticmethod
    def _estimate_tokens(messages: Sequence[ChatMessage]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, str):
                total += len(content) // 4 + 4
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            total += len(part.get("text") or "") // 4 + 4
                        elif part.get("type") == "image_url":
                            total += 256
            images = msg.get("images")
            if images and isinstance(images, (list, tuple)):
                total += len(images) * 256
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                import json

                total += len(json.dumps(tool_calls)) // 4
        return total

    @staticmethod
    def _group_turns(non_system_messages: Sequence[ChatMessage]) -> list[list[ChatMessage]]:
        turns: list[list[ChatMessage]] = []
        current_turn: list[ChatMessage] = []
        for msg in non_system_messages:
            if msg.get("role") == "user" and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)
        return turns

    def _prune_history(self) -> None:
        if self.strategy != "sliding_window" or not self.messages:
            return

        has_system = self.messages[0].get("role") == "system"
        system_msg = [self.messages[0]] if has_system else []
        non_system = self.messages[1:] if has_system else self.messages[:]

        if not non_system:
            return

        turns = self._group_turns(non_system)

        if self.max_turns is not None and len(turns) > self.max_turns:
            turns = turns[-self.max_turns :]

        if self.max_tokens_budget is not None:
            while (
                len(turns) > 1
                and self._estimate_tokens(system_msg + [m for t in turns for m in t]) > self.max_tokens_budget
            ):
                turns.pop(0)

        self.messages = system_msg + [m for t in turns for m in t]

    def _prepare_messages(self, user_message: ChatMessage) -> list[ChatMessage]:
        if self.strategy != "sliding_window":
            return [*self.messages, user_message]

        candidate = [*self.messages, user_message]
        has_system = candidate[0].get("role") == "system"
        system_msg = [candidate[0]] if has_system else []
        non_system = candidate[1:] if has_system else candidate[:]

        turns = self._group_turns(non_system)

        if self.max_turns is not None and len(turns) > self.max_turns:
            turns = turns[-self.max_turns :]

        if self.max_tokens_budget is not None:
            while (
                len(turns) > 1
                and self._estimate_tokens(system_msg + [m for t in turns for m in t]) > self.max_tokens_budget
            ):
                turns.pop(0)

        return system_msg + [m for t in turns for m in t]

    def _commit(self, user_message: ChatMessage, content: str, tool_calls: Sequence[ToolCall]) -> None:
        self.messages.append(user_message)
        assistant: ChatMessage = {"role": "assistant", "content": content}
        if tool_calls:
            assistant["tool_calls"] = list(tool_calls)
        self.messages.append(assistant)
        self._prune_history()

    @overload
    def send(
        self,
        user_input: str,
        *,
        images: Sequence[str | bytes | Any] | None = None,
        stream: Literal[True],
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> Iterator[str]: ...

    @overload
    def send(
        self,
        user_input: str,
        *,
        images: Sequence[str | bytes | Any] | None = None,
        stream: Literal[False] = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatResponse: ...

    @overload
    def send(
        self,
        user_input: str,
        *,
        images: Sequence[str | bytes | Any] | None = None,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatResponse | Iterator[str]: ...

    def send(
        self,
        user_input: str,
        *,
        images: Sequence[str | bytes | Any] | None = None,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatResponse | Iterator[str]:
        self._claim_turn()
        user_message: ChatMessage = {"role": "user", "content": user_input}
        if images:
            user_message["images"] = list(images)
        messages = self._prepare_messages(user_message)
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

    @overload
    async def asend(
        self,
        user_input: str,
        *,
        images: Sequence[str | bytes | Any] | None = None,
        stream: Literal[True],
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> AsyncIterator[str]: ...

    @overload
    async def asend(
        self,
        user_input: str,
        *,
        images: Sequence[str | bytes | Any] | None = None,
        stream: Literal[False] = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatResponse: ...

    @overload
    async def asend(
        self,
        user_input: str,
        *,
        images: Sequence[str | bytes | Any] | None = None,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatResponse | AsyncIterator[str]: ...

    async def asend(
        self,
        user_input: str,
        *,
        images: Sequence[str | bytes | Any] | None = None,
        stream: bool = False,
        json_mode: bool = False,
        response_model: type[Any] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ChatResponse | AsyncIterator[str]:
        self._claim_turn()
        user_message: ChatMessage = {"role": "user", "content": user_input}
        if images:
            user_message["images"] = list(images)
        messages = self._prepare_messages(user_message)
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

    @property
    def estimated_tokens(self) -> int:
        """Returns the approximate token count of the current conversation history."""
        return self._estimate_tokens(self.messages)

    @property
    def turn_count(self) -> int:
        """Returns the number of user turns currently stored in history."""
        return sum(1 for m in self.messages if m.get("role") == "user")

    def clear(self) -> None:
        """Clears conversation history, preserving the initial system prompt if one was provided."""
        with self._state_lock:
            if self._busy:
                raise RuntimeError("Cannot clear session while a request is in progress.")
            self.messages = []
            if self.system_prompt:
                self.messages.append({"role": "system", "content": self.system_prompt})

    def reset(self, system_prompt: str | None = None) -> None:
        """Resets the session completely, optionally setting a new system prompt."""
        with self._state_lock:
            if self._busy:
                raise RuntimeError("Cannot reset session while a request is in progress.")
            self.system_prompt = system_prompt
            self.messages = []
            if system_prompt:
                self.messages.append({"role": "system", "content": system_prompt})

    def act(
        self,
        user_input: str,
        *,
        tools: Sequence[Callable[..., Any] | dict[str, Any]],
        max_steps: int = 5,
        images: Sequence[str | bytes | Any] | None = None,
        **options: Any,
    ) -> ChatResponse:
        """Executes an autonomous tool-calling loop: sends prompt, executes tool calls,

        feeds observations back to the model, and returns the final answer.
        """
        from omnillm.core.tools import ToolRegistry

        registry = ToolRegistry(tools)
        response = self.send(
            user_input,
            images=images,
            tools=registry.schemas,
            stream=False,
            **options,
        )

        step = 1
        while response.tool_calls and step < max_steps:
            step += 1
            for call in response.tool_calls:
                func_info = call.get("function", {})
                name = func_info.get("name", "") if isinstance(func_info, dict) else ""
                args = func_info.get("arguments", "{}") if isinstance(func_info, dict) else "{}"
                call_id = call.get("id") or f"call_{step}"
                result_str = registry.execute(name, args)
                tool_msg: ChatMessage = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": result_str,
                }
                self.messages.append(tool_msg)

            self._claim_turn()
            try:
                next_result = self.manager.chat(
                    self.backend,
                    self.model,
                    self.messages,
                    tools=registry.schemas,
                    stream=False,
                    **options,
                )
                if not isinstance(next_result, ChatResponse):
                    raise RuntimeError("Backend returned a stream during autonomous tool execution.")
                assistant_msg: ChatMessage = {"role": "assistant", "content": next_result.content}
                if next_result.tool_calls:
                    assistant_msg["tool_calls"] = list(next_result.tool_calls)
                self.messages.append(assistant_msg)
                self._prune_history()
                response = next_result
            finally:
                self._release_turn()

        return response

    async def aact(
        self,
        user_input: str,
        *,
        tools: Sequence[Callable[..., Any] | dict[str, Any]],
        max_steps: int = 5,
        images: Sequence[str | bytes | Any] | None = None,
        **options: Any,
    ) -> ChatResponse:
        """Executes an asynchronous autonomous tool-calling loop."""
        from omnillm.core.tools import ToolRegistry

        registry = ToolRegistry(tools)
        response = await self.asend(
            user_input,
            images=images,
            tools=registry.schemas,
            stream=False,
            **options,
        )

        step = 1
        while response.tool_calls and step < max_steps:
            step += 1
            for call in response.tool_calls:
                func_info = call.get("function", {})
                name = func_info.get("name", "") if isinstance(func_info, dict) else ""
                args = func_info.get("arguments", "{}") if isinstance(func_info, dict) else "{}"
                call_id = call.get("id") or f"call_{step}"
                result_str = await registry.aexecute(name, args)
                tool_msg: ChatMessage = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": result_str,
                }
                self.messages.append(tool_msg)

            self._claim_turn()
            try:
                next_result = await self.manager.achat(
                    self.backend,
                    self.model,
                    self.messages,
                    tools=registry.schemas,
                    stream=False,
                    **options,
                )
                if not isinstance(next_result, ChatResponse):
                    raise RuntimeError("Backend returned a stream during autonomous tool execution.")
                assistant_msg: ChatMessage = {"role": "assistant", "content": next_result.content}
                if next_result.tool_calls:
                    assistant_msg["tool_calls"] = list(next_result.tool_calls)
                self.messages.append(assistant_msg)
                self._prune_history()
                response = next_result
            finally:
                self._release_turn()

        return response
