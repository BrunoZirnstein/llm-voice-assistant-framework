import webrtcvad

from .IVoiceActivityDetector import IVoiceActivityDetector


class WebRTCVAD(IVoiceActivityDetector):
    def __init__(self, aggressiveness: int = 3, sample_rate: int = 16000):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate

    def is_speech(self, audio_frame: bytes) -> bool:
        return self.vad.is_speech(audio_frame, self.sample_rate)
