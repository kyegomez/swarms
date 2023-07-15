from swarms.tools.agent_tools import *
from pydantic import ValidationError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Boss Node ----------
class BossNode:
    """
    The BossNode class is responsible for creating and executing tasks using the BabyAGI model.
    It takes a language model (llm), a vectorstore for memory, an agent_executor for task execution, and a maximum number of iterations for the BabyAGI model.
    """
    def __init__(self, llm, vectorstore, agent_executor, max_iterations):
        if not llm or not vectorstore or not agent_executor or not max_iterations:
            logging.error("llm, vectorstore, agent_executor, and max_iterations cannot be None.")
            raise ValueError("llm, vectorstore, agent_executor, and max_iterations cannot be None.")
        self.llm = llm
        self.vectorstore = vectorstore
        self.agent_executor = agent_executor
        self.max_iterations = max_iterations

        try:
            self.baby_agi = BabyAGI.from_llm(
                llm=self.llm,
                vectorstore=self.vectorstore,
                task_execution_chain=self.agent_executor,
                max_iterations=self.max_iterations,
            )
        except ValidationError as e:
            logging.error(f"Validation Error while initializing BabyAGI: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected Error while initializing BabyAGI: {e}")
            raise

    def create_task(self, objective):
        """
        Creates a task with the given objective.
        """
        if not objective:
            logging.error("Objective cannot be empty.")
            raise ValueError("Objective cannot be empty.")
        return {"objective": objective}

    def execute_task(self, task):
        """
        Executes a task using the BabyAGI model.
        """
        if not task:
            logging.error("Task cannot be empty.")
            raise ValueError("Task cannot be empty.")
        try:
            self.baby_agi(task)
        except Exception as e:
            logging.error(f"Error while executing task: {e}")
            raise