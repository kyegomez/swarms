class PromptRefiner:

    def __init__(self, system_prompt: str, llm):
        super().__init__()
        self.system_prompt = system_prompt
        self.llm = llm

    def run(self, task: str):
        refine = self.llm(
            f"System Prompt: {self.system_prompt} Current task: {task}")
        return refine
