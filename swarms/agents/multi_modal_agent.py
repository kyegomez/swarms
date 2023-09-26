# from swarms.workers.multi_modal_workers.multi_modal_agent import MultiModalVisualAgent
from swarms.workers.multi_modal_workers.multi_modal_agent import MultiModalVisualAgent

class MultiModalVisualAgent:
    def __init__(self, agent: MultiModalVisualAgent):
        self.agent = agent
    
    def _run(self, text: str) -> str:
        #run the multi-modal visual agent with the give task
        return self.agent.run_text(text)

