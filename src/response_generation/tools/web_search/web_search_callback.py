import os

from pydub import AudioSegment
from pydub.playback import play


def web_search_callback():
    audio = AudioSegment.from_file(os.path.join("sounds", "web_search.wav"))
    play(audio)
