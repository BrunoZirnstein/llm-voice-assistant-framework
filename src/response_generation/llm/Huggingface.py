from typing import Optional

from dotenv import load_dotenv
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain_core.language_models import LLM
from langchain_core.language_models.chat_models import BaseChatModel

from .BaseLLMProvider import BaseLLMProvider

load_dotenv(".env")


class Huggingface(BaseLLMProvider):
    def __init__(self) -> None:
        pass

    def llm(
        self, repo_id: str, task: Optional[str] = "text-generation", model_kwargs: Optional[dict] = None
    ) -> BaseChatModel:
        llm = HuggingFaceHub(repo_id=repo_id, task=task, model_kwargs=model_kwargs)
        return ChatHuggingFace(llm=llm)
