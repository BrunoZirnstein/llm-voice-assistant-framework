import logging

import pyaudio

from .IAudioListener import IAudioListener

logger = logging.getLogger(__name__)


class Porcupine_Listener(IAudioListener):
    def __init__(self, sample_rate: int, frame_length: int):
        self.sample_rate = sample_rate
        self.frame_length = frame_length

        pyaudio_instance = pyaudio.PyAudio()

        self.audio_stream = pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            frames_per_buffer=self.frame_length,
            input=True,
        )

    def fetch_audio_frame(self) -> bytes:
        return self.audio_stream.read(self.frame_length, exception_on_overflow=False)
