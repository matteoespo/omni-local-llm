import ollama
from omnillm.core.base import LLMBackend

class OllamaAdapter(LLMBackend):
    def pull_model(self, model_name: str, **kwargs):
        print(f"[omnillm -> Ollama] Checking if '{model_name}' is available locally...")
        try:
            ollama.show(model_name)
            print(f"[omnillm -> Ollama] '{model_name}' is already available.")
        except ollama.ResponseError as e:
            if e.status_code == 404:
                print(f"[omnillm -> Ollama] Model not found. Pulling '{model_name}'... This might take a while.")
                ollama.pull(model_name)
                print("[omnillm -> Ollama] Pull complete!")
            else:
                raise e

    def chat(self, model_name: str, messages: list, stream: bool = False, json_mode: bool = False, **kwargs):
        self.pull_model(model_name) 
        print("[omnillm -> Ollama] Generating response...")
        
        chat_kwargs = {"model": model_name, "messages": messages, "stream": stream}
        if json_mode:
            chat_kwargs["format"] = "json"
            
        response = ollama.chat(**chat_kwargs)
        
        if stream:
            def generator():
                for chunk in response:
                    yield chunk['message'].get('content', '')
            return generator()
        else:
            return response['message']['content']

    async def achat(self, model_name: str, messages: list, stream: bool = False, json_mode: bool = False, **kwargs):
        import asyncio
        from ollama import AsyncClient
        
        await asyncio.to_thread(self.pull_model, model_name)
        
        print("[omnillm -> Ollama Async] Generating response...")
        client = AsyncClient()
        chat_kwargs = {"model": model_name, "messages": messages, "stream": stream}
        if json_mode:
            chat_kwargs["format"] = "json"
            
        response = await client.chat(**chat_kwargs)
        
        if stream:
            async def async_generator():
                async for chunk in response:
                    yield chunk['message'].get('content', '')
            return async_generator()
        else:
            return response['message']['content']