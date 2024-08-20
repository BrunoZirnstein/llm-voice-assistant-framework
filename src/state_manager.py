from enum import Enum
from logging import Logger


class State(Enum):
    IMPORT_AUDIO_DETECTION = "Importing Audio Detection modules"
    IMPORT_TRANSCRIPTION = "Importing Transcription modules"
    IMPORT_LLMS = "Importing LLM modules"
    IMPORT_AUDIO_GENERATION = "Importing Audio Generation modules"

    SETUP_AUDIO_DETECTION = "Setting up Audio Detection Pipeline"
    SETUP_TRANSCRIPTION = "Setting up Transcription Service"
    SETUP_LLM_TOOLS = "Setting up LLM Tools"
    SETUP_LLM_AGENT = "Setting up LLM Agent"
    SETUP_LLM = "Setting up Large Language Model"
    SETUP_AUDIO_GENERATION = "Setting up Audio Generation Service"

    LISTENING_FOR_ACTIVATION = "Assistant active"

    KEYWORD_DETECTED = "Keyword detected"
    VOICE_DETECTED = "Voice detected"
    LONG_SILENCE_DETECTED = "Long silence detected"
    WAITING_TIME_EXCEEDED = "Waiting time for voice exceeded"

    TRANSCRIPTION_IN_PROGRESS = "Calling Transcription Service"
    TRANSCRIPTION_SUCCESS = "Transcription successful"
    TRANSCRIPTION_NOTHING_DETECTED = "Nothing detected"

    RESPONSE_GENERATION_IN_PROGRESS = "Generating response"
    RESPONSE_GENERATION_SUCCESS = "Response generated"

    PLAYING_RESPONSE = "Speaking"

    REQUEST_FINISHED = "Request finished"

    TERMINATED = "Terminated application"


class StateManager:
    def __init__(self):
        self.state = None

    def set_state(self, state: State, logger: Logger, **kwargs):
        logger.info(state.value)
