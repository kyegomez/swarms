import os
from loguru import logger
import pygame
import requests
import tempfile
from openai import OpenAI


class OpenAITTS:
    """
    A class to interact with OpenAI API and play the generated audio with improved streaming capabilities.
    """

    def __init__(self, *args, **kwargs):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), *args, **kwargs
        )
        pygame.init()

    def run(
        self, task: str, play_sound: bool = True, *args, **kwargs
    ):
        """
        Run a task with the OpenAI API and optionally play the generated audio with improved streaming.

        Args:
            task (str): The task to be executed.
            play_sound (bool): If True, play the generated audio.

        Returns:
            None
        """
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=task,
                *args,
                **kwargs,
            )
            audio_url = response["url"]
            logger.info("Task completed successfully.")

            if play_sound:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3"
                ) as tmp_file:
                    with requests.get(audio_url, stream=True) as r:
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                    pygame.mixer.music.load(tmp_file.name)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
        except Exception as e:
            logger.error(f"Error during task execution: {str(e)}")


# client = OpenAITTS(api_key=os.getenv("OPENAI_API_KEY"))
# client.run("Hello world! This is a streaming test.", play_sound=True)


def text_to_speech(
    task: str, play_sound: bool = True, *args, **kwargs
):
    out = OpenAITTS().run(
        task, play_sound=play_sound, *args, **kwargs
    )
    return out


# print(text_to_speech(task="hello"))
