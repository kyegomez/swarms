from typing import List
from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.utils.loguru_logger import logger


class SocietyOfAgents(BaseSwarm):
    def __init__(
        self,
        name: str = None,
        description: str = None,
        agents: List[Agent] = None,
        max_loops: int = 1,
        rules: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.conversation = Conversation(
            time_enabled=True, rules=rules, *args, **kwargs
        )

    def run(self, task: str = None, *args, **kwargs):
        loop = 0

        try:
            while loop < self.max_loops:
                for agent in self.agents:
                    out = agent.run(task, *args, **kwargs)

                    # Save the conversation
                    self.conversation.add(agent.agent_name, out)

                    task = out

                    # Log the agent's output
                    logger.info(f"Agent {agent.agent_name} output: {out}")

                loop += 1

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

        return out
