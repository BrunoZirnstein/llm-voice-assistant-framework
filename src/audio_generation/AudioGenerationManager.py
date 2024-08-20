import pyaudio

from .IAudioGenerator import IAudioGenerator


class AudioGenerationManager:
    def __init__(self, audio_generator: IAudioGenerator):
        self.audio_generator = audio_generator
        self.pa_instance = pyaudio.PyAudio()
        self.stream = self.pa_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.audio_generator.get_sample_rate(),
            output=True,
        )

    def generate_audio(self, text: str):
        return self.audio_generator.generate_audio(text)

    def play(self, audio_data):
        self.stream.start_stream()
        num_frames = len(audio_data) // 2
        self.stream.write(audio_data, num_frames)
        self.stream.stop_stream()

    def close(self):
        self.stream.close()
        self.pa_instance.terminate()
