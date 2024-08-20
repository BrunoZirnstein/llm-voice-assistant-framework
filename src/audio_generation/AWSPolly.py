import boto3

from .IAudioGenerator import IAudioGenerator


class AWSPolly(IAudioGenerator):
    def __init__(self, language_code: str, model_name: str):
        self.client = boto3.client("polly")
        self.language_code = language_code
        self.model_name = model_name

    def get_sample_rate(self) -> int:
        return 16000

    def generate_audio(self, text: str):
        response = self.client.synthesize_speech(
            LanguageCode=self.language_code,
            OutputFormat="pcm",
            Text=text,
            VoiceId=self.model_name,
        )

        audio_data = response["AudioStream"].read()
        return audio_data
