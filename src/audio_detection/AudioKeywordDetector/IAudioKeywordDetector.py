from abc import ABC, abstractmethod


class IAudioKeywordDetector(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def detect_keyword(self, audio_frame: bytes) -> bool:
        pass
