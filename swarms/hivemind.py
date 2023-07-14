# many boss + workers in unison
#kye gomez jul 13 4:01pm, can scale up the number of swarms working on a probkem with `hivemind(swarms=4, or swarms=auto which will scale the agents depending on the complexity)`

import concurrent.futures
import logging

from swarms.swarms import Swarms
from swarms.tools.agent_tools import *

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



class HiveMind:
    def __init__(self, openai_api_key="", num_swarms=1, max_workers=None):
        self.openai_api_key = openai_api_key
        self.num_swarms = num_swarms
        self.swarms = [Swarms(openai_api_key) for _ in range(num_swarms)]
        self.vectorstore = self.initialize_vectorstore()
        self.max_workers = max_workers if max_workers else min(32, num_swarms)

    def initialize_vectorstore(self):
        try:
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)
            return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            raise

    def run_swarm(self, swarm, objective):
        try:
            return swarm.run_swarms(objective)
        except Exception as e:
            logging.error(f"An error occurred in run_swarms: {e}")

    def run_swarms(self, objective, timeout=None):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.run_swarm, swarm, objective) for swarm in self.swarms}
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                try:
                    results.append(future.result())
                except Exception as e:
                    logging.error(f"An error occurred in a swarm: {e}")
        return results