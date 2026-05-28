from omnillm.core.manager import LocalLLMManager

class ChatSession:
    def __init__(self, manager: LocalLLMManager, backend: str, model: str, system_prompt: str = None):
        self.manager = manager
        self.backend = backend
        self.model = model
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def send(self, user_input: str, stream: bool = False, json_mode: bool = False, **kwargs):
        self.messages.append({"role": "user", "content": user_input})
        response = self.manager.chat(self.backend, self.model, self.messages, stream=stream, json_mode=json_mode, **kwargs)
        
        if stream:
            def stream_wrapper():
                full_content = ""
                for chunk in response:
                    full_content += chunk
                    yield chunk
                self.messages.append({"role": "assistant", "content": full_content})
            return stream_wrapper()
        else:
            self.messages.append({"role": "assistant", "content": response})
            return response

    async def asend(self, user_input: str, stream: bool = False, json_mode: bool = False, **kwargs):
        self.messages.append({"role": "user", "content": user_input})
        response = await self.manager.achat(self.backend, self.model, self.messages, stream=stream, json_mode=json_mode, **kwargs)
        
        if stream:
            async def async_stream_wrapper():
                full_content = ""
                async for chunk in response:
                    full_content += chunk
                    yield chunk
                self.messages.append({"role": "assistant", "content": full_content})
            return async_stream_wrapper()
        else:
            self.messages.append({"role": "assistant", "content": response})
            return response
