from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from . import AgentPipelineManager, LLMPipelineManager
from .agent import BaseAgentProvider
from .llm import BaseLLMProvider


class ResponseGenerationPipelineManager:
    def __init__(
        self, prompt: ChatPromptTemplate, llm: Union[BaseLLMProvider, BaseAgentProvider], tools: List[BaseTool]
    ):
        if isinstance(llm, BaseAgentProvider):
            assert tools is not None
            self.pipeline = AgentPipelineManager(
                llm, prompt, tools, llm.convert_tool, llm.format_function_messages, llm.output_parser
            )
        elif isinstance(llm, BaseLLMProvider):
            self.pipeline = LLMPipelineManager(llm, prompt)
        else:
            raise ValueError("llm must be of type BaseLLMProvider or BaseAgentProvider")

    async def generate_response(self, user_query: str):
        async for chunk in self.pipeline.run(user_query):
            yield chunk
