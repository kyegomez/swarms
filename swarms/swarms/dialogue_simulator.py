import os
from typing import Callable, List


class DialogueSimulator:
    """
    Dialogue Simulator
    ------------------

    Args:
    ------
    agents: List[Callable]
    max_iters: int
    name: str

    Usage:
    ------
    >>> from swarms import DialogueSimulator
    >>> from swarms.structs.agent import Agent
    >>> agents = Agent()
    >>> agents1 = Agent()
    >>> model = DialogueSimulator([agents, agents1], max_iters=10, name="test")
    >>> model.run("test")
    """

    def __init__(
        self,
        agents: List[Callable],
        max_iters: int = 10,
        name: str = None,
    ):
        self.agents = agents
        self.max_iters = max_iters
        self.name = name

    def run(self, message: str = None):
        """Run the dialogue simulator"""
        try:
            step = 0
            if self.name and message:
                prompt = f"Name {self.name} and message: {message}"
                for agent in self.agents:
                    agent.run(prompt)
                step += 1

            while step < self.max_iters:
                speaker_idx = step % len(self.agents)
                speaker = self.agents[speaker_idx]
                speaker_message = speaker.run(prompt)

                for receiver in self.agents:
                    message_history = (
                        f"Speaker Name: {speaker.name} and message:"
                        f" {speaker_message}"
                    )
                    receiver.run(message_history)

                print(f"({speaker.name}): {speaker_message}")
                print("\n")
                step += 1
        except Exception as error:
            print(f"Error running dialogue simulator: {error}")

    def __repr__(self):
        return (
            f"DialogueSimulator({self.agents}, {self.max_iters},"
            f" {self.name})"
        )

    def save_state(self):
        """Save the state of the dialogue simulator"""
        try:
            if self.name:
                filename = f"{self.name}.txt"
                with open(filename, "w") as file:
                    file.write(str(self))
        except Exception as error:
            print(f"Error saving state: {error}")

    def load_state(self):
        """Load the state of the dialogue simulator"""
        try:
            if self.name:
                filename = f"{self.name}.txt"
                with open(filename, "r") as file:
                    return file.read()
        except Exception as error:
            print(f"Error loading state: {error}")

    def delete_state(self):
        """Delete the state of the dialogue simulator"""
        try:
            if self.name:
                filename = f"{self.name}.txt"
                os.remove(filename)
        except Exception as error:
            print(f"Error deleting state: {error}")
