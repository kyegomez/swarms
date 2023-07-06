from swarms.tools.agent_tools import *
from pydantic import ValidationError

# ---------- Boss Node ----------
class BossNode:
    def __init__(self, llm, vectorstore, agent_executor, max_iterations):
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
            print(f"Validation Error while initializing BabyAGI: {e}")
        except Exception as e:
            print(f"Unexpected Error while initializing BabyAGI: {e}")
