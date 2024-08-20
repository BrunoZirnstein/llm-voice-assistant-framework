import logging
import time
from typing import Callable, List, Type

import speech_recognition as sr

from state_manager import State, StateManager

from .AudioKeywordDetector import IAudioKeywordDetector
from .AudioListener import IAudioListener
from .VoiceActivityDetector import IVoiceActivityDetector, SileroVAD, WebRTCVAD

logger = logging.getLogger(__name__)


class AudioDetectionManager:
    """
    Orchestrates the identification and transcription process by initializing and managing operations
    of audio listeners, keyword detectors, and audio transcribers. It uses these objects to manage
    an audio processing pipeline in the `process_audio_stream` method.

    Args:
        listener (IAudioListener): An object that implements the 'IAudioListener' interface to listen to audio.
        detector (IAudioKeywordDetector): An object that implements the 'IAudioKeywordDetector' interface to detect keyword in audio.
        transcriber (IAudioTranscriber): An object that implements the 'IAudioTranscriber' interface to transcribe audio data.
        transcription_callback (Callable[[str], None]): Function to be invoked after a successful transcription occurs. It must accept a single argument - the transcribed string.
        vad (IVoiceActivityDetector): An object that detects presence of voice in the audio frame

    Raises:
        TypeError: Raises an exception when listener, detector, transcriber, vad attributes do not match their respective interfaces.
        TypeError: Raises an exception when `transcription_callback` is not a callable object.

    Methods:
        process_audio_stream(): Initiates the audio processing pipeline, continuously listens to audio,
                                detects keywords and, if detected, transcribes the remainder. Invokes the `transcription_callback` function with the transcription.
    """

    KEYWORDS_FRAME_SIZE: int = 512
    VAD_FRAME_SIZE: int = 320  # frame size for WebRTC VAD
    # VAD_FRAME_SIZE: int = 512 # frame size for SileroVAD
    SAMPLE_RATE: int = 16000

    VOICE_START_MAX_WAIT_TIME_S = 5  # Amount of time to wait for user to speak after keyword detection in seconds
    SILENCE_DURATION_IN_MS: int = 500

    frame_duration = (VAD_FRAME_SIZE * 1.0) / SAMPLE_RATE
    max_silence_frames = int(SILENCE_DURATION_IN_MS / 1000 / frame_duration)
    frame_length = max(VAD_FRAME_SIZE, KEYWORDS_FRAME_SIZE)

    def __init__(
        self,
        listener_cls: Type[IAudioListener],
        detector_cls: Type[IAudioKeywordDetector],
        recorded_audio_callback: Callable[[sr.AudioData], None],
        state_manager: StateManager,
        vad_cls: Type[IVoiceActivityDetector] = WebRTCVAD,  # SileroVAD,
    ) -> None:
        if not issubclass(listener_cls, IAudioListener):
            raise TypeError("Listener must be a subclass of IAudioListener.")
        if not issubclass(detector_cls, IAudioKeywordDetector):
            raise TypeError("Detector must be a subclass of IAudioKeywordDetector.")
        if not issubclass(vad_cls, IVoiceActivityDetector):
            raise TypeError("Voice Activity Detector (VAD) must be a subclass of IVoiceActivityDetector.")

        if not callable(recorded_audio_callback):
            raise TypeError("recorded_audio_callback must be callable.")

        self.listener: IAudioListener = listener_cls(sample_rate=self.SAMPLE_RATE, frame_length=self.frame_length)
        self.detector: IAudioKeywordDetector = detector_cls()
        self.recorded_audio_callback: Callable[[sr.AudioData], None] = recorded_audio_callback
        self.state_manager: StateManager = state_manager
        self.vad: IVoiceActivityDetector = vad_cls(sample_rate=self.SAMPLE_RATE)

    def process_audio_stream(self) -> None:
        """
        Runs the audio processing pipeline indefinitely.

        The pipeline works as follows:
        1. Fetches a new audio chunk from the listener.
        2. Detects if a keyword is present in the audio chunk.
        3. If a keyword is detected, it waits for voice activity.
        4. Records the user's speech until a silence longer than a specified duration is detected.
        5. Once the user stops speaking, it invokes the 'recorded_audio_callback' function with the recorded audio data.
        6. Finally, resets the state variables for the next round of keyword detection and speech recording.

        If no voice activity is detected within a specified duration after keyword detection, it assumes a false alarm and resets the state variables.
        """

        found_keyword: bool = False
        is_speaking: bool = False
        audio_data: List[bytes] = []
        silence_frames: int = 0
        last_voice_detected_timestamp: float = 0.0

        while True:
            # Fetch and process an audio chunk
            audio_chunk: bytes = self.listener.fetch_audio_frame()
            audio_chunk_length: int = len(audio_chunk)

            for i in range(0, audio_chunk_length, self.KEYWORDS_FRAME_SIZE):
                end_index: int = 2 * (i + self.KEYWORDS_FRAME_SIZE)
                if end_index > audio_chunk_length:
                    break  # Skip incomplete frames

                keywords_frame: bytes = audio_chunk[2 * i : end_index]

                # If a keyword is detected, start or continue waiting for voice
                if self.detector.detect_keyword(keywords_frame):
                    # Keyword detected, start waiting for voice
                    self.state_manager.set_state(State.KEYWORD_DETECTED, logger)
                    found_keyword = True
                    last_voice_detected_timestamp = time.time()

                # Detect voice in all frames if keyword found
                if found_keyword:
                    for j in range(i, end_index, self.VAD_FRAME_SIZE):
                        vad_end_index: int = 2 * (j + self.VAD_FRAME_SIZE)

                        if vad_end_index <= audio_chunk_length:
                            vad_frame: bytes = audio_chunk[2 * j : vad_end_index]
                            if self.vad.is_speech(vad_frame):
                                if not is_speaking:
                                    self.state_manager.set_state(State.VOICE_DETECTED, logger)
                                is_speaking = True
                                silence_frames = 0
                                last_voice_detected_timestamp = time.time()
                            else:
                                silence_frames += 1
                            audio_data.append(keywords_frame)

                            # If Voice has stopped, pass the audio data to the callback function
                            if is_speaking and silence_frames >= self.max_silence_frames:
                                self.state_manager.set_state(State.LONG_SILENCE_DETECTED, logger)

                                speech_end_index = len(audio_data) - self.max_silence_frames
                                clean_audio_data = audio_data[:speech_end_index]

                                audio: sr.AudioData = sr.AudioData(
                                    b"".join(clean_audio_data), sample_rate=self.SAMPLE_RATE, sample_width=2
                                )
                                self.recorded_audio_callback(audio)

                                # Reset states for next detection
                                found_keyword = False
                                is_speaking = False
                                audio_data[:] = []
                                silence_frames = 0

                # Time to wait after keyword detection is exceeded.
                if found_keyword and time.time() - last_voice_detected_timestamp > self.VOICE_START_MAX_WAIT_TIME_S:
                    self.state_manager.set_state(State.WAITING_TIME_EXCEEDED, logger)

                    # Reset states for next detection
                    found_keyword = False
                    is_speaking = False
                    audio_data[:] = []
                    silence_frames = 0
