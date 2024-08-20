import logging

logger = logging.getLogger(__name__)

import asyncio
import os
import time

import speech_recognition as sr

from state_manager import State, StateManager


class App:
    def __init__(self) -> None:
        self.state_manager = StateManager()

        self.state_manager.set_state(State.IMPORT_AUDIO_DETECTION, logger)
        from audio_detection import AudioDetectionManager
        from audio_detection.AudioKeywordDetector import Porcupine_Picovoice
        from audio_detection.AudioListener import Porcupine_Listener

        self.state_manager.set_state(State.IMPORT_TRANSCRIPTION, logger)
        from audio_transcription.AudioTranscriber import (
            GoogleCloudSpeech,
            OpenAI_Whisper,
            VoskAPI,
        )

        self.state_manager.set_state(State.IMPORT_LLMS, logger)
        from response_generation import (
            ResponseGenerationPipelineManager,
            agent_prompt,
            chat_prompt,
        )
        from response_generation.agent import OpenAIAgent

        # from response_generation.llm import Huggingface, Ollama, OpenAI
        from response_generation.tools.web_search import (  # TavilyAPI, YouAPI,
            AzureBingAPIv7,
        )

        self.state_manager.set_state(State.IMPORT_AUDIO_GENERATION, logger)
        from audio_generation import AudioGenerationManager, AWSPolly, GoogleCloudTTS

        self.state_manager.set_state(State.SETUP_AUDIO_DETECTION, logger)
        # Audio Detection
        self.audio_detection_manager = AudioDetectionManager(
            Porcupine_Listener, Porcupine_Picovoice, self.recorded_audio_callback, state_manager=self.state_manager
        )

        self.state_manager.set_state(State.SETUP_TRANSCRIPTION, logger)
        # self.transcriber = VoskAPI()
        # self.transcriber = GoogleCloudSpeech()
        self.transcriber = OpenAI_Whisper()

        # Response Generation

        self.state_manager.set_state(State.SETUP_LLM_TOOLS, logger)
        web_search = AzureBingAPIv7()
        tools = [web_search]

        self.state_manager.set_state(State.SETUP_LLM_AGENT, logger)
        # agent = OpenAIAgent(model_name="gpt-4-0125-preview")
        agent = OpenAIAgent(model_name="gpt-4o")

        # self.state_manager.set_state(State.SETUP_LLM)

        # openai = OpenAI()
        # llm = openai.llm(model_name="gpt-3.5-turbo-0125")
        # llm = openai.llm(model_name="gpt-4-0125-preview")

        # huggingface = Huggingface()
        # llm = huggingface.llm(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1")
        # llm = huggingface.llm(repo_id="HuggingFaceH4/zephyr-7b-beta")

        # ollama = Ollama()
        # llm = ollama.llm("llama2")

        # self.llm_pipeline_manager = ResponseGenerationPipelineManager(chat_prompt, llm)
        self.llm_pipeline_manager = ResponseGenerationPipelineManager(agent_prompt, agent, tools)

        self.state_manager.set_state(State.SETUP_AUDIO_GENERATION, logger)
        # Audio Generation
        audio_generator = GoogleCloudTTS(language_code="de-DE", model_name="de-DE-Wavenet-B")
        # audio_generator = AWSPolly(language_code="de-DE", model_name="Daniel")
        self.audio_generation_manager = AudioGenerationManager(audio_generator)

        if not os.path.isfile(os.path.join("sounds", "web_search.wav")):
            search_web_audio = self.audio_generation_manager.generate_audio("Lass mich kurz im Internet nachschauen.")

            if not os.path.isdir("sounds"):
                os.makedirs("sounds")

            with open(os.path.join("sounds", "web_search.wav"), "wb") as f:
                f.write(search_web_audio)

    def __del__(self):
        try:
            self.audio_generation_manager.close()
        except AttributeError:
            pass

        self.state_manager.set_state(State.TERMINATED, logger)

    def main(self):
        self.state_manager.set_state(State.LISTENING_FOR_ACTIVATION, logger)
        self.audio_detection_manager.process_audio_stream()

    def recorded_audio_callback(self, audio_data: sr.AudioData) -> None:
        self.total_time_start = time.time()

        self.state_manager.set_state(State.TRANSCRIPTION_IN_PROGRESS, logger)

        start = time.time()
        transcription_result: str = self.transcriber.transcribe(audio_data)
        logger.info("Transcription took %s seconds", time.time() - start)

        if transcription_result:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.transcription_callback(transcription_result))
        else:
            self.state_manager.set_state(State.TRANSCRIPTION_NOTHING_DETECTED, logger)

    async def transcription_callback(self, transcription: str) -> None:
        logger.info("> %s", transcription)

        start = time.time()

        response = ""
        async for chunk in self.llm_pipeline_manager.generate_response(transcription):
            response += chunk

        logger.info(response)
        logger.info("Response Generation took %s seconds", time.time() - start)

        self.generate_audio(response)

    def generate_audio(self, text: str):
        start = time.time()
        audio_data = self.audio_generation_manager.generate_audio(text)
        logger.info("Autio Generation took %s seconds", time.time() - start)

        logger.info("Total  pipeline took %s seconds", time.time() - self.total_time_start)

        self.state_manager.set_state(State.PLAYING_RESPONSE, logger)
        self.audio_generation_manager.play(audio_data)
        self.state_manager.set_state(State.REQUEST_FINISHED, logger)


if __name__ == "__main__":
    # set up and config logging
    logging_level = os.environ.get("LOGLEVEL", "INFO").upper()
    if logging_level not in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]:
        logging_level = "INFO"
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging_level)

    logger.info("Drinking coffee ☕️")

    app = App()
    app.main()
