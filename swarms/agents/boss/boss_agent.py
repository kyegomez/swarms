from swarms.tools.agent_tools import *

# ---------- Boss Node ----------
class BossNode:
    def __init__(self, llm, vectorstore, task_execution_chain, verbose, max_iterations):
        self.llm = llm
        self.vectorstore = vectorstore
        self.task_execution_chain = task_execution_chain
        self.verbose = verbose
        self.max_iterations = max_iterations

        self.baby_agi = BabyAGI.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            task_execution_chain=self.task_execution_chain
        )

    def create_task(self, objective):
        return {"objective": objective}

    def execute_task(self, task):
        self.baby_agi(task)