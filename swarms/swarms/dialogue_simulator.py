from typing import List
from swarms.workers.worker import Worker


class DialogueSimulator:
    def __init__(
        self,
        agents: List[Worker],
        selection_func
    ):
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_func
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
        
    def start(self, name: str, message: str):
        #init conv with a message from name
        prompt = f"Name {name} and message: {message}"

        for agent in self.agents:
            agent.run(prompt)
        
        #increment time
        self._step += 1
    
    def step(self) -> tuple[str, str]:
        #choose next speaker
        speaker_idx = self.select_next_speaker(
            self._step,
            self.agent
        )
        speaker = self.agents[speaker_idx]

        #2. next speaker ends message
        message = speaker.send()

        #everyone receives messages
        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        
        #increment time
        self._step += 1

        return speaker.name, message
    
    def select_next_speaker(step: int, agents) -> int:
        idx = (step) % len(agents)
        return idx