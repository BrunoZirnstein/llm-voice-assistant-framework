from abc import ABC, abstractmethod
from typing import Optional


class IAudioTranscriber(ABC):
    @abstractmethod
    def __init__(self, sample_rate: int):
        pass

    @abstractmethod
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        pass
