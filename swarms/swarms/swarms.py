import asyncio
import logging
from typing import Optional

from langchain import OpenAI

from swarms.boss.boss_node import BossNode
from swarms.workers.worker_node import WorkerNode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Constants ----------
ROOT_DIR = "./data/"

class HierarchicalSwarm:
    def __init__(
        self, 
        openai_api_key: Optional[str] = "", 
        use_vectorstore: Optional[bool] = True, 
        use_async: Optional[bool] = True, 
        worker_name: Optional[str] = "Swarm Worker AI Assistant",
        verbose: Optional[bool] = False,
        human_in_the_loop: Optional[bool] = True, 
        boss_prompt: Optional[str] = "You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n",
        worker_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.5,
        max_iterations: Optional[int] = None,
        logging_enabled: Optional[bool] = True
    ):
        self.openai_api_key = openai_api_key
        self.use_vectorstore = use_vectorstore
        self.use_async = use_async
        self.worker_name = worker_name
        self.human_in_the_loop = human_in_the_loop
        self.boss_prompt = boss_prompt
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.logging_enabled = logging_enabled
        self.verbose = verbose

        self.worker_node = WorkerNode(
            openai_api_key=self.openai_api_key,
            temperature=self.temperature,
            human_in_the_loop=self.human_in_the_loop,
            verbose=self.verbose
        )

        self.boss_node = BossNode(
            api_key=self.openai_api_key,
            worker_node=self.worker_node,
            llm_class=OpenAI,
            max_iterations=self.max_iterations,
            verbose=self.verbose
        )

        self.logger = logging.getLogger()
        if not logging_enabled:
            self.logger.disabled = True

    def run(self, objective):
        try:
            self.boss_node.task = self.boss_node.create_task(objective)
            logging.info(f"Running task: {self.boss_node.task}")
            if self.use_async:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(self.boss_node.run())
            else:
                result = self.boss_node.run()
            logging.info(f"Completed tasks: {self.boss_node.task}")
            return result
        except Exception as e:
            logging.error(f"An error occurred in run: {e}")
            return None

def swarm(
    api_key: Optional[str]="", 
    objective: Optional[str]=""
):
    if not api_key or not isinstance(api_key, str):
        logging.error("Invalid OpenAI key")
        raise ValueError("A valid OpenAI API key is required")
    if not objective or not isinstance(objective, str):
        logging.error("Invalid objective")
        raise ValueError("A valid objective is required")
    try:
        swarms = HierarchicalSwarm(api_key, use_async=False)
        result = swarms.run(objective)
        if result is None:
            logging.error("Failed to run swarms")
        else:
            logging.info(f"Successfully ran swarms with results: {result}")
        return result
    except Exception as e:
        logging.error(f"An error occured in swarm: {e}")
        return None
    