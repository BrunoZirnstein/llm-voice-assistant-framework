from dotenv import load_dotenv
from langchain_core.language_models import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from .BaseLLMProvider import BaseLLMProvider

load_dotenv(".env")


class Ollama(BaseLLMProvider):
    def __init__(self) -> None:
        pass

    def llm(self, model: str) -> BaseChatModel:
        return OllamaFunctions(model=model)
