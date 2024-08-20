from abc import ABC, abstractmethod
from typing import List


class IVoiceActivityDetector(ABC):
    @abstractmethod
    def __init__(self, aggressiveness: int, sample_rate: int):
        pass

    @abstractmethod
    def is_speech(self, audio_frame: List[int]) -> bool:
        pass
