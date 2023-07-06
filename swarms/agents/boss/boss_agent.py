from swarms.tools.agent_tools import *
from pydantic import ValidationError

# ---------- Boss Node ----------
class BossNode:
    def __init__(self, llm, vectorstore, task_execution_chain, verbose, max_iterations):
        self.llm = llm
        self.vectorstore = vectorstore
        self.task_execution_chain = task_execution_chain
        self.verbose = verbose
        self.max_iterations = max_iterations

        try:
            # Ensure llm is a dictionary before passing it to BabyAGI
            assert isinstance(llm, dict), "llm should be a dictionary."

            self.baby_agi = BabyAGI.from_llm(
                llm=self.llm,
                vectorstore=self.vectorstore,
                task_execution_chain=self.task_execution_chain,
                verbose=self.verbose,
                max_iterations=self.max_iterations,
            )
        except ValidationError as e:
            print(f"Validation Error while initializing BabyAGI: {e}")
        except Exception as e:
            print(f"Unexpected Error while initializing BabyAGI: {e}")

    def create_task(self, objective):
        try:
            task = {"objective": objective}
            return task
        except Exception as e:
            print(f"Unexpected Error while creating a task: {e}")

    def execute_task(self, task):
        try:
            self.baby_agi(task)
        except Exception as e:
            print(f"Unexpected Error while executing a task: {e}")