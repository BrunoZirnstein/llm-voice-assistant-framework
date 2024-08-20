from dotenv import load_dotenv
from langchain_core.language_models import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .BaseLLMProvider import BaseLLMProvider

load_dotenv(".env")


class OpenAI(BaseLLMProvider):
    def __init__(self) -> None:
        pass

    def llm(self, model_name: str) -> BaseChatModel:
        return ChatOpenAI(model=model_name, temperature=0)
