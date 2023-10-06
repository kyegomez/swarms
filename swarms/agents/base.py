class AbsractAgent:
    def __init__(
        self,
        llm,
        temperature
    ) -> None:
        pass
    
    #single query
    def run(self, task: str):
        pass

    # # conversational back and forth
    # def chat(self, message: str):
    #     message_historys = []
    #     message_historys.append(message)

    #     reply = self.run(message)
    #     message_historys.append(reply)

    #     return message_historys

    # def step(self, message):
    #     pass

    # def reset(self):
    #     pass
