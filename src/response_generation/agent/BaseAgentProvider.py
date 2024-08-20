from abc import ABC, abstractmethod
from typing import List

from langchain_core.tools import BaseTool


class BaseAgentProvider(ABC):
    @abstractmethod
    def __init__(self):
        pass
