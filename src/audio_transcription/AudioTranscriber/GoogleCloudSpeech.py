import logging
from typing import Optional

import speech_recognition as sr

from .IAudioTranscriber import IAudioTranscriber

logger = logging.getLogger(__name__)


class GoogleCloudSpeech(IAudioTranscriber):
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def transcribe(self, audio_data: sr.AudioData) -> Optional[str]:
        try:
            transcription = self.recognizer.recognize_google_cloud(audio_data, language="de-DE")
        except sr.UnknownValueError:
            return None

        return transcription
