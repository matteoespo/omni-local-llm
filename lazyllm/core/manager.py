from lazyllm.adapters.ollama_adapter import OllamaAdapter
from lazyllm.adapters.llamacpp_adapter import LlamaCPPAdapter

class LocalLLMManager:
    def __init__(self):
        # Register available backends here
        self.backends = {
            "ollama": OllamaAdapter(),
            "llama.cpp": LlamaCPPAdapter()
        }

    def chat(self, backend: str, model: str, messages: list, **kwargs) -> str:
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' is not supported. Choose from {list(self.backends.keys())}")
        
        adapter = self.backends[backend]
        return adapter.chat(model_name=model, messages=messages, **kwargs)