import wave
from typing import Optional
from swarms.models.base_llm import AbstractLLM
from abc import ABC, abstractmethod


class BaseTTSModel(AbstractLLM):
    def __init__(
        self,
        model_name,
        voice,
        chunk_size,
        save_to_file: bool = False,
        saved_filepath: Optional[str] = None,
    ):
        self.model_name = model_name
        self.voice = voice
        self.chunk_size = chunk_size

    def save(self, filepath: Optional[str] = None):
        pass

    def load(self, filepath: Optional[str] = None):
        pass

    @abstractmethod
    def run(self, task: str, *args, **kwargs):
        pass

    def save_to_file(self, speech_data, filename):
        """Save the speech data to a file.

        Args:
            speech_data (bytes): The speech data.
            filename (str): The path to the file where the speech will be saved.
        """
        with wave.open(filename, "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(22050)
            file.writeframes(speech_data)
