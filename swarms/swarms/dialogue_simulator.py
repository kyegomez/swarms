from typing import List
from swarms.workers.worker import Worker


class DialogueSimulator:
    """
    Dialogue Simulator
    ------------------

    Args:
    ------


    Usage:
    --------



    """

    def __init__(self, agents: List[Worker]):
        self.agents = agents

    def run(self, max_iters: int, name: str = None, message: str = None):
        step = 0
        if name and message:
            prompt = f"Name {name} and message: {message}"
            for agent in self.agents:
                agent.run(prompt)
            step += 1

        while step < max_iters:
            speaker_idx = step % len(self.agents)
            speaker = self.agents[speaker_idx]
            speaker_message = speaker.run(prompt)

            for receiver in self.agents:
                message_history = (
                    f"Speaker Name: {speaker.name} and message: {speaker_message}"
                )
                receiver.run(message_history)

            print(f"({speaker.name}): {speaker_message}")
            print("\n")
            step += 1
