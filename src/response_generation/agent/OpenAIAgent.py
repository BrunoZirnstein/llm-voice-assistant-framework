from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_tool

from response_generation import BaseResponseGenerationPipeline
from response_generation.llm.OpenAI import OpenAI

from . import BaseAgentProvider


class OpenAIAgent(BaseAgentProvider):
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125") -> BaseResponseGenerationPipeline:
        openai = OpenAI()
        self.llm = openai.llm(model_name=model_name)

        self.convert_tool = convert_to_openai_tool
        self.format_function_messages = format_to_openai_tool_messages
        self.output_parser = OpenAIToolsAgentOutputParser()
