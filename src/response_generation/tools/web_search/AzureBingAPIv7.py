import logging
import os
import threading
from typing import Any, Callable, Optional, Type

import requests
from dotenv import load_dotenv
from langchain_core.tools import BaseModel, BaseTool, Field

from .web_search_callback import web_search_callback

load_dotenv(".env")

logger = logging.getLogger(__name__)

AZURE_BING_SEARCH_KEY = os.environ.get("AZURE_BING_SEARCH_KEY")
if not AZURE_BING_SEARCH_KEY:
    raise Exception("AZURE_BING_SEARCH_KEY not set in .env file. See .env.template for reference.")

AZURE_BING_SEARCH_ENDPOINT = os.environ.get("AZURE_BING_SEARCH_ENDPOINT")
if not AZURE_BING_SEARCH_ENDPOINT:
    raise Exception("AZURE_BING_SEARCH_ENDPOINT not set in .env file. See .env.template for reference.")


class AzureBingAPIv7Input(BaseModel):
    """ """

    query: str = Field(description="Text query that is being searched for. The query MUST be in German.")


class AzureBingAPIv7(BaseTool):
    args_schema: Type[BaseModel] = AzureBingAPIv7Input
    name: str = "web_search"
    description: str = "Search the web for a text query."

    def _run(self, query: str, **kwargs: Any) -> Any:
        logger.info(f"Searching for '{query}'")

        callback_thread = threading.Thread(target=web_search_callback)
        callback_thread.start()

        headers = {"Ocp-Apim-Subscription-Key": AZURE_BING_SEARCH_KEY}
        endpoint = AZURE_BING_SEARCH_ENDPOINT + "/v7.0/search"

        params = {"q": query}
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()

        search_results = response.json()["webPages"]["value"]
        filtered_search_results = [
            {"rank": int(result["id"][-1]) + 1, "url": result["url"], "snippet": result["snippet"]}
            for result in search_results
        ]
        return filtered_search_results
