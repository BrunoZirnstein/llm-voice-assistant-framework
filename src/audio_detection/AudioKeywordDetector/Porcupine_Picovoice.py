import os

import numpy as np
import pvporcupine
from dotenv import load_dotenv

from .IAudioKeywordDetector import IAudioKeywordDetector

load_dotenv()

PORCUPINE_ACCESS_KEY = os.environ.get("PORCUPINE_ACCESS_KEY")
if not PORCUPINE_ACCESS_KEY:
    raise Exception("PORCUPINE_ACCESS_KEY not set in .env file. See .env.template for reference.")

models_dir = os.path.join("models", "Porcupine_Picovoice")
porcupine_params = os.path.join(models_dir, "porcupine_params_de.pv")
fridolin_model = os.path.join(models_dir, "Fridolin_de_mac_v3_0_0.ppn")


class Porcupine_Picovoice(IAudioKeywordDetector):
    def __init__(self):
        self.handle = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            model_path=porcupine_params,
            keyword_paths=[fridolin_model],
        )

    def __del__(self):
        self.handle.delete()

    def detect_keyword(self, audio_frame: bytes) -> bool:
        is_keyword_detected = self.handle.process(np.frombuffer(audio_frame, dtype=np.int16))
        return is_keyword_detected >= 0
