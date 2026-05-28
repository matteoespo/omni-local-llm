from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from omnillm.core.base import LLMBackend

class LlamaCPPAdapter(LLMBackend):
    def __init__(self):
        # Add state to keep track of the currently loaded model
        self.active_model_name = None
        self.llm_instance = None

    def pull_model(self, model_name: str, **kwargs):
        local_path = hf_hub_download(repo_id=model_name, filename=kwargs.get("filename"))
        return local_path

    def chat(self, model_name: str, messages: list, stream: bool = False, json_mode: bool = False, **kwargs):
        local_model_path = self.pull_model(model_name, **kwargs)
        
        # only load if it's a new model
        if self.active_model_name != model_name:
            print(f"[omnillm -> llama.cpp] Loading '{model_name}' into RAM/VRAM...")
            
            self.llm_instance = Llama(
                model_path=local_model_path, 
                n_gpu_layers=-1, 
                verbose=False
            )
            self.active_model_name = model_name
        else:
            print(f"[omnillm -> llama.cpp] '{model_name}' is already in memory. Chatting...")
        
        chat_kwargs = {"messages": messages, "stream": stream}
        if json_mode:
            chat_kwargs["response_format"] = {"type": "json_object"}
            
        response = self.llm_instance.create_chat_completion(**chat_kwargs)
        
        if stream:
            def generator():
                for chunk in response:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        yield delta['content']
            return generator()
        else:
            return response['choices'][0]['message']['content']

    async def achat(self, model_name: str, messages: list, stream: bool = False, json_mode: bool = False, **kwargs):
        import asyncio
        
        sync_result = await asyncio.to_thread(
            self.chat, model_name, messages, stream=stream, json_mode=json_mode, **kwargs
        )
        
        if stream:
            async def async_generator():
                loop = asyncio.get_running_loop()
                while True:
                    try:
                        chunk = await loop.run_in_executor(None, next, sync_result)
                        yield chunk
                    except StopIteration:
                        break
            return async_generator()
        else:
            return sync_result