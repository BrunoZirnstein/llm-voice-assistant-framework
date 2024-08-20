import logging
import os
from typing import Any, Type

import requests
from dotenv import load_dotenv
from langchain_core.tools import BaseModel, BaseTool, Field
from tavily import TavilyClient

load_dotenv(".env")

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise Exception("TAVILY_API_KEY not set in .env file. See .env.template for reference.")


class TavilyAPIInput(BaseModel):
    """ """

    query: str = Field(description="Text query that is being searched for")


class TavilyAPI(BaseTool):
    args_schema: Type[BaseModel] = TavilyAPIInput
    name: str = "web_search"
    description: str = "Search the web for a text query."

    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    def _run(self, query: str, **kwargs: Any) -> Any:
        logger.info(f"Searching for '{query}'")

        response = self.tavily.search(query=query, search_depth="advanced")
        return [{"url": obj["url"], "content": obj["content"]} for obj in response["results"]]
