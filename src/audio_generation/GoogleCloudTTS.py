import io

from google.cloud import texttospeech

from .IAudioGenerator import IAudioGenerator


class GoogleCloudTTS(IAudioGenerator):
    def __init__(self, language_code: str = "en-US", model_name: str = "en-US-Journey-D", speed: float = 1.2):
        self.client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=model_name)
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=speed
        )

    def get_sample_rate(self) -> int:
        return 24000

    def generate_audio(self, text: str):
        input = texttospeech.SynthesisInput(text=text)
        response = self.client.synthesize_speech(input=input, voice=self.voice, audio_config=self.audio_config)
        audio_data = io.BytesIO(response.audio_content).getvalue()
        return audio_data
