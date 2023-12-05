import logging
from typing import Union

from pegasus import Pegasus


class PegasusEmbedding:
    """
    Pegasus

    Args:
        modality (str): Modality to use for embedding
        multi_process (bool, optional): Whether to use multi-process. Defaults to False.
        n_processes (int, optional): Number of processes to use. Defaults to 4.

    Usage:
    --------------
    pegasus = PegasusEmbedding(modality="text")
    pegasus.embed("Hello world")


    vision
    --------------
    pegasus = PegasusEmbedding(modality="vision")
    pegasus.embed("https://i.imgur.com/1qZ0K8r.jpeg")

    audio
    --------------
    pegasus = PegasusEmbedding(modality="audio")
    pegasus.embed("https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav")



    """

    def __init__(
        self,
        modality: str,
        multi_process: bool = False,
        n_processes: int = 4,
    ):
        self.modality = modality
        self.multi_process = multi_process
        self.n_processes = n_processes
        try:
            self.pegasus = Pegasus(
                modality, multi_process, n_processes
            )
        except Exception as e:
            logging.error(
                "Failed to initialize Pegasus with modality:"
                f" {modality}: {e}"
            )
            raise

    def embed(self, data: Union[str, list[str]]):
        """Embed the data"""
        try:
            return self.pegasus.embed(data)
        except Exception as e:
            logging.error(
                f"Failed to generate embeddings. Error: {e}"
            )
            raise
