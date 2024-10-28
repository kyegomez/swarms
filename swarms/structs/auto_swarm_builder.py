from typing import List, Any, Dict, Optional


class AutoSwarmBuilder:
    def __init__(self, task: str, num_agents: int = 10, batch_size: int = 10):
        self.task = task
        self.num_agents = num_agents
        self.batch_size = batch_size

    def run(self, task: str, image_url: str = None, *args, **kwargs):
        pass

    def _create_swarm(self):
        pass

    def _create_agents(self):
        pass

    def _run_agents(self):
        pass
