import json
import logging
from typing import Optional

import speech_recognition as sr
from vosk import KaldiRecognizer, Model

from .IAudioTranscriber import IAudioTranscriber

logger = logging.getLogger(__name__)


class VoskAPI(IAudioTranscriber):
    def __init__(self):
        self.model = Model(model_name="vosk-model-de-0.21")

    def transcribe(self, audio_data: sr.AudioData) -> Optional[str]:
        rec = KaldiRecognizer(self.model, audio_data.sample_rate)
        data = audio_data.frame_data
        rec.AcceptWaveform(data)
        raw_result = rec.FinalResult()
        return json.loads(raw_result).get("text", "")
