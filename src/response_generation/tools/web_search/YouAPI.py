import json
import logging
import os
from typing import Any, Type

import requests
from dotenv import load_dotenv
from langchain_community.utilities.you import YouSearchAPIWrapper
from langchain_core.tools import BaseModel, BaseTool, Field

load_dotenv(".env")

logger = logging.getLogger(__name__)

YOU_API_KEY = os.environ.get("YOU_API_KEY")
if not YOU_API_KEY:
    raise Exception("YOU_API_KEY not set in .env file. See .env.template for reference.")


class YouAPIInput(BaseModel):
    """ """

    query: str = Field(description="Text query that is being searched for")


class YouAPI(BaseTool):
    args_schema: Type[BaseModel] = YouAPIInput
    name: str = "web_search"
    description: str = "Search the web for a text query."

    you = YouSearchAPIWrapper(ydc_api_key=YOU_API_KEY, num_web_results=1)

    def _run(self, query: str, **kwargs: Any) -> Any:
        logger.info(f"Searching for '{query}'")

        response = self.you.raw_results(query)
        return response["hits"]


if __name__ == "__main__":
    you = YouAPI()
    print(you._run(query="Wettervorhersage für Berlin am 14. Februar 2024"))
