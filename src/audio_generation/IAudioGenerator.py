from abc import ABC, abstractmethod


class IAudioGenerator(ABC):
    @abstractmethod
    def __init__(self, language_code: str, model_name: str):
        pass

    @abstractmethod
    def generate_audio(self, text: str) -> bytes:
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        pass
