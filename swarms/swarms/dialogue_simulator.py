from typing import List
from swarms.workers.worker import Worker

class DialogueSimulator:
    def __init__(self, agents: List[Worker]):
        self.agents = agents

    def run(
        self, 
        max_iters: int, 
        name: str = None, 
        message: str = None
    ):
        step = 0
        if name and message:
            prompt = f"Name {name} and message: {message}"
            for agent in self.agents:
                agent.run(prompt)
            step += 1

        while step < max_iters:
            speaker_idx = step % len(self.agents)
            speaker = self.agents[speaker_idx]
            speaker_message = speaker.run()
            for receiver in self.agents:
                receiver.receive(speaker.name, speaker_message)
            print(f"({speaker.name}): {speaker_message}")
            print("\n")
            step += 1