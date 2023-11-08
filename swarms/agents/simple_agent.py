from termcolor import colored


class SimpleAgent:
    """
    Simple Agent is a simple agent that runs a flow.

    Args:
        name (str): Name of the agent
        flow (Flow): Flow to run

    Example:
    >>> from swarms.agents.simple_agent import SimpleAgent
    >>> from swarms.structs import Flow
    >>> from swarms.models import OpenAIChat
    >>> api_key = ""
    >>> llm = OpenAIChat()

    """

    def __init__(
        self,
        name: str,
        flow,
    ):
        self.name = name
        self.flow = flow
        self.message_history = []

    def run(self, task: str) -> str:
        """Run method"""
        metrics = print(
            colored(f"Agent {self.name} is running task: {task}", "red"))
        print(metrics)

        response = self.flow.run(task)
        self.message_history.append((self.name, response))
        return response
