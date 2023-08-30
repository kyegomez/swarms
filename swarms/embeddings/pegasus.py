import logging
from typing import Union
from pegasus import Pegasus

# import oceandb
# from oceandb.utils.embedding_functions import MultiModalEmbeddingfunction


class PegasusEmbedding:
    def __init__(self, modality: str, multi_process: bool = False, n_processes: int = 4):
        self.modality = modality
        self.multi_process = multi_process
        self.n_processes = n_processes
        try:
            self.pegasus = Pegasus(modality, multi_process, n_processes)
        except Exception as e:
            logging.error(f"Failed to initialize Pegasus with modality: {modality}: {e}")
            raise
    
    def embed(self, data: Union[str, list[str]]):
        try:
            return self.pegasus.embed(data)
        except Exception as e:
            logging.error(f"Failed to generate embeddings. Error: {e}")
            raise

