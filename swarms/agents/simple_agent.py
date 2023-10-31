class SimpleAgent:
    def __init__(
        self,
        name: str,
        llm,
    ):
        self.name = name
        self.llm = llm
        self.message_history = []

    def run(self, task: str) -> str:
        response = self.model(task)
        self.message_history.append((self.name, response))
        return response
