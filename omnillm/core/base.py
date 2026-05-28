from abc import ABC, abstractmethod

class LLMBackend(ABC):
    @abstractmethod
    def pull_model(self, model_name: str, **kwargs):
        """Downloads the model if it doesn't exist locally."""
        pass

    @abstractmethod
    def chat(self, model_name: str, messages: list, stream: bool = False, json_mode: bool = False, tools: list = None, **kwargs):
        """Sends a prompt to the model and returns the response (or a generator if stream=True)."""
        pass

    @abstractmethod
    async def achat(self, model_name: str, messages: list, stream: bool = False, json_mode: bool = False, tools: list = None, **kwargs):
        """Asynchronous version of chat."""
        pass