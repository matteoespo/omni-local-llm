from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import time

from omnillm.core.manager import LocalLLMManager

app = FastAPI(title="Omni-LLM OpenAI API", version="0.1.0")
manager = LocalLLMManager()

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ResponseFormat(BaseModel):
    type: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    response_format: Optional[ResponseFormat] = None
    temperature: Optional[float] = 0.7
    tools: Optional[List[Dict[str, Any]]] = None

def parse_model_string(model_string: str):
    """Parses 'backend/model_name' into backend and model. Defaults to ollama."""
    parts = model_string.split('/', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "ollama", model_string

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    backend, model_name = parse_model_string(request.model)
    
    messages_dict = []
    for m in request.messages:
        d = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls is not None:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id is not None:
            d["tool_call_id"] = m.tool_call_id
        messages_dict.append(d)
    
    json_mode = False
    if request.response_format and request.response_format.type == "json_object":
        json_mode = True

    try:
        response = await manager.achat(
            backend=backend,
            model=model_name,
            messages=messages_dict,
            stream=request.stream,
            json_mode=json_mode,
            tools=request.tools,
            temperature=request.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    created_time = int(time.time())

    if request.stream:
        async def stream_generator():
            async for chunk in response:
                chunk_data = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        message_dict = {"role": "assistant"}
        if isinstance(response, dict):
            message_dict["content"] = response.get("content", "")
            if response.get("tool_calls"):
                message_dict["tool_calls"] = response["tool_calls"]
        else:
            message_dict["content"] = response

        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": "tool_calls" if message_dict.get("tool_calls") else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
