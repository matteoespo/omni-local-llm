from abc import ABC, abstractmethod

class LLMBackend(ABC):
    @abstractmethod
    def pull_model(self, model_name: str, **kwargs):
        """Downloads the model if it doesn't exist locally."""
        pass

    @abstractmethod
    def chat(self, model_name: str, messages: list, **kwargs) -> str:
        """Sends a prompt to the model and returns the response."""
        pass