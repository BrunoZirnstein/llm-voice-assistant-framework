import logging
from typing import Optional

import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI

from .IAudioTranscriber import IAudioTranscriber

logger = logging.getLogger(__name__)

load_dotenv(".env")


class OpenAI_Whisper(IAudioTranscriber):
    def __init__(self):
        self.client = OpenAI()

    def transcribe(self, audio_data: sr.AudioData) -> Optional[str]:
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1", file=("audio.wav", audio_data.get_wav_data(), "audio/wav")
        )
        return transcript.text
