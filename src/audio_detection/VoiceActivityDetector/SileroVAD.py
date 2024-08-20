import contextlib
import logging
import os

import numpy as np
import torch

from .IVoiceActivityDetector import IVoiceActivityDetector

logger = logging.getLogger(__name__)


class SileroVAD(IVoiceActivityDetector):
    def __init__(self, sample_rate: int = 16000):
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True, onnx=False
            )
        logger.debug("Model and utils loaded from snakers4/silero-vad")

        self.model = model
        self.sample_rate = sample_rate
        self.window_size_samples = 512 if sample_rate == 16000 else 256
        self.model.reset_states()

    def is_speech(self, audio_frame: bytes, probability_threshold: float = 0.8) -> bool:
        audio_array = np.frombuffer(audio_frame, dtype=np.int16)
        audio_array = np.copy(audio_array)
        audio_tensor = torch.from_numpy(audio_array).float()

        if len(audio_array) != self.window_size_samples:
            raise ValueError(f"Expected {self.window_size_samples} samples, but got {len(audio_array)} samples.")

        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
        return speech_prob > probability_threshold
