import json
import time
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from omnillm.core.errors import (
    BackendNotFoundError,
    BackendUnavailableError,
    InvalidRequestError,
    UnsupportedFeatureError,
)
from omnillm.core.manager import LocalLLMManager
from omnillm.core.types import ChatChunk, ChatResponse


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    response_format: ResponseFormat | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    n: int = Field(default=1, ge=1)


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Parses 'backend/model_name' into a registered backend and model name."""
    if "/" not in model_string:
        return "ollama", model_string
    backend, model = model_string.split("/", 1)
    if not backend or not model:
        raise InvalidRequestError("Model must be 'backend/model' or a non-empty Ollama model name.")
    return backend, model


def _error_response(error: Exception) -> JSONResponse:
    status_code = 500
    error_type = "server_error"
    if isinstance(error, BackendNotFoundError):
        status_code, error_type = 404, "invalid_request_error"
    elif isinstance(error, BackendUnavailableError):
        status_code, error_type = 503, "server_error"
    elif isinstance(error, (InvalidRequestError, UnsupportedFeatureError, ValueError)):
        status_code, error_type = 400, "invalid_request_error"
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": str(error), "type": error_type}},
    )


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def create_app(manager: LocalLLMManager | None = None) -> FastAPI:
    app = FastAPI(title="Omni-Local-LLM OpenAI API", version="0.2.0")
    llm_manager = manager or LocalLLMManager()

    @app.get("/v1/models", response_model=None)
    async def list_models() -> JSONResponse | dict[str, Any]:
        try:
            models = [
                {
                    "id": f"{backend}/{model}",
                    "object": "model",
                    "created": 0,
                    "owned_by": backend,
                }
                for backend, model in llm_manager.list_models()
            ]
        except Exception as error:
            return _error_response(error)
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest) -> JSONResponse | StreamingResponse | dict[str, Any]:
        if request.n != 1:
            return _error_response(InvalidRequestError("Only n=1 is supported."))

        try:
            backend, model_name = parse_model_string(request.model)
            response = await llm_manager.achat(
                backend=backend,
                model=model_name,
                messages=[message.model_dump(exclude_none=True) for message in request.messages],
                stream=request.stream,
                json_mode=request.response_format is not None and request.response_format.type == "json_object",
                tools=request.tools,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop,
            )
        except Exception as error:
            return _error_response(error)

        chat_id = f"chatcmpl-{uuid4().hex}"
        created = int(time.time())

        if request.stream:
            if not isinstance(response, AsyncIterator):
                return _error_response(InvalidRequestError("Backend did not return a stream."))

            async def stream() -> AsyncIterator[str]:
                finish_reason = "stop"
                yield _sse(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
                try:
                    async for chunk in response:
                        if not isinstance(chunk, ChatChunk):
                            raise InvalidRequestError("Backend returned an invalid stream chunk.")
                        delta: dict[str, Any] = {}
                        if chunk.content:
                            delta["content"] = chunk.content
                        if chunk.tool_calls:
                            delta["tool_calls"] = list(chunk.tool_calls)
                            finish_reason = "tool_calls"
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if delta:
                            yield _sse(
                                {
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": request.model,
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                                }
                            )
                    yield _sse(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        }
                    )
                except Exception as error:
                    payload = bytes(_error_response(error).body).decode()
                    yield f"data: {payload}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        if not isinstance(response, ChatResponse):
            return _error_response(InvalidRequestError("Backend returned a stream."))

        message: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            message["tool_calls"] = list(response.tool_calls)
        payload: dict[str, Any] = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": response.finish_reason,
                }
            ],
        }
        if response.usage is not None and (usage := response.usage.as_openai()) is not None:
            payload["usage"] = usage
        return payload

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
