import ollama
from lazyllm.core.base import LLMBackend

class OllamaAdapter(LLMBackend):
    def pull_model(self, model_name: str, **kwargs):
        print(f"[lazyllm -> Ollama] Checking if '{model_name}' is available locally...")
        try:
            ollama.show(model_name)
            print(f"[lazyllm -> Ollama] '{model_name}' is already available.")
        except ollama.ResponseError as e:
            if e.status_code == 404:
                print(f"[lazyllm -> Ollama] Model not found. Pulling '{model_name}'... This might take a while.")
                ollama.pull(model_name)
                print("[lazyllm -> Ollama] Pull complete!")
            else:
                raise e

    def chat(self, model_name: str, messages: list, **kwargs) -> str:
        self.pull_model(model_name) 
        print(f"[lazyllm -> Ollama] Generating response...")
        response = ollama.chat(model=model_name, messages=messages)
        return response['message']['content']