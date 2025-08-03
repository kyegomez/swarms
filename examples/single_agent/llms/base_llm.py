from swarms.structs.agent import Agent


class BaseLLM:
    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] = [],
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def run(self, task: str, *args, **kwargs):
        pass

    def __call__(self, task: str, *args, **kwargs):
        return self.run(task, *args, **kwargs)


agent = Agent(
    llm=BaseLLM(),
    agent_name="BaseLLM",
    system_prompt="You are a base LLM agent.",
)
