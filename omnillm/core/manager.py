from omnillm.adapters.ollama_adapter import OllamaAdapter
from omnillm.adapters.llamacpp_adapter import LlamaCPPAdapter

class LocalLLMManager:
    def __init__(self):
        # Register available backends here
        self.backends = {
            "ollama": OllamaAdapter(),
            "llama.cpp": LlamaCPPAdapter()
        }

    def chat(self, backend: str, model: str, messages: list, stream: bool = False, json_mode: bool = False, tools: list = None, **kwargs):
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' is not supported. Choose from {list(self.backends.keys())}")
        
        adapter = self.backends[backend]
        return adapter.chat(model_name=model, messages=messages, stream=stream, json_mode=json_mode, tools=tools, **kwargs)

    async def achat(self, backend: str, model: str, messages: list, stream: bool = False, json_mode: bool = False, tools: list = None, **kwargs):
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' is not supported. Choose from {list(self.backends.keys())}")
        
        adapter = self.backends[backend]
        return await adapter.achat(model_name=model, messages=messages, stream=stream, json_mode=json_mode, tools=tools, **kwargs)

    def create_session(self, backend: str, model: str, system_prompt: str = None):
        from omnillm.core.session import ChatSession
        return ChatSession(manager=self, backend=backend, model=model, system_prompt=system_prompt)