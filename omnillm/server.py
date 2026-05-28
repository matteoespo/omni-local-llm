from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import time

from omnillm.core.manager import LocalLLMManager

app = FastAPI(title="Omni-LLM OpenAI API", version="0.1.0")
manager = LocalLLMManager()

class ChatMessage(BaseModel):
    role: str
    content: str

class ResponseFormat(BaseModel):
    type: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    response_format: Optional[ResponseFormat] = None
    temperature: Optional[float] = 0.7

def parse_model_string(model_string: str):
    """Parses 'backend/model_name' into backend and model. Defaults to ollama."""
    parts = model_string.split('/', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "ollama", model_string

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    backend, model_name = parse_model_string(request.model)
    
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    
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
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
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
