from abc import ABC, abstractmethod

from langchain_core.language_models import LLM
from langchain_core.language_models.chat_models import BaseChatModel


class BaseLLMProvider(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def llm(self) -> BaseChatModel:
        pass
