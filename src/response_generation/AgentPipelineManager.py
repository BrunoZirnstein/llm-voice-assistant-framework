import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from langchain.agents import AgentExecutor
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool

from . import BaseResponseGenerationPipeline
from .agent import BaseAgentProvider

logger = logging.getLogger(__name__)


class AgentPipelineManager:
    def __init__(
        self,
        llm: BaseAgentProvider,
        prompt: ChatPromptTemplate,
        tools: Optional[List[BaseTool]],
        convert_tool: Callable[[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]], Dict[str, Any]],
        format_function_messages: Callable[[Sequence[Tuple[AgentAction, str]]], List[BaseMessage]],
        output_parser: AgentOutputParser,
    ):
        self.chat_history = []
        llm_with_tools = llm.llm.bind(tools=[convert_tool(t) for t in tools])

        callback_manager = None

        agent = (
            RunnablePassthrough.assign(agent_scratchpad=lambda x: format_function_messages(x["intermediate_steps"]))
            | prompt
            | llm_with_tools
            | output_parser
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, callback_manager=callback_manager)

    async def run(self, user_query: str):
        result = self.agent_executor.invoke({"input": user_query, "chat_history": self.chat_history})
        response = result["output"]
        yield response

        self.chat_history += [HumanMessage(content=user_query), AIMessage(content=response)]
