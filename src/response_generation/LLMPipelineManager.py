import logging

from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AIMessage, HumanMessage
from langchain_core.language_models import LLM
from langchain_core.output_parsers import BaseLLMOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(".env")
logger = logging.getLogger(__name__)


class LLMPipelineManager:
    def __init__(
        self,
        chat_prompt: ChatPromptTemplate,
        llm: LLM,
        output_parser: BaseLLMOutputParser = StrOutputParser(),
    ):
        self.chat_history = []
        self.llm = llm

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.chat_prompt = chat_prompt
        self.output_parser = output_parser

        self.agent = self.chat_prompt | self.llm | self.output_parser

    async def run(self, user_query: str):
        ai_message = ""
        async for chunk in self.agent_executor.astream({"input": user_query, "chat_history": self.chat_history}):
            yield chunk
            ai_message += chunk

        self.chat_history += [HumanMessage(content=user_query), AIMessage(content=ai_message)]
