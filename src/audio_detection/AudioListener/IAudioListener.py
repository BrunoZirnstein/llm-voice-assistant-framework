from abc import ABC, abstractmethod


class IAudioListener(ABC):
    @abstractmethod
    def __init__(self, sample_rate: int, frame_length: int):
        pass

    @abstractmethod
    def fetch_audio_frame(self) -> bytes:
        pass
