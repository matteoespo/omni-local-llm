from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from lazyllm.core.base import LLMBackend

class LlamaCPPAdapter(LLMBackend):
    def pull_model(self, model_name: str, **kwargs):
        filename = kwargs.get("filename")
        if not filename:
            raise ValueError("llama.cpp requires a 'filename' kwarg (e.g., 'model.q4_k_m.gguf')")

        print(f"[lazyllm -> llama.cpp] Ensuring '{filename}' from repo '{model_name}' is downloaded...")
        local_path = hf_hub_download(repo_id=model_name, filename=filename)
        return local_path

    def chat(self, model_name: str, messages: list, **kwargs) -> str:
        local_model_path = self.pull_model(model_name, **kwargs)
        
        print(f"[lazyllm -> llama.cpp] Loading model into memory...")
        # verbose=False keeps the llama.cpp C-level logs from flooding the terminal
        llm = Llama(model_path=local_model_path, verbose=False)
        
        print(f"[lazyllm -> llama.cpp] Generating response...")
        response = llm.create_chat_completion(messages=messages)
        return response['choices'][0]['message']['content']