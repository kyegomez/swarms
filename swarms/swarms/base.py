from abc import ABC, abstractmethod

class AbstractSwarm(ABC):
    # TODO: Pass in abstract LLM class that can utilize Hf or Anthropic models, Move away from OPENAI
    # TODO: ADD Universal Communication Layer, a ocean vectorstore instance
    # TODO: BE MORE EXPLICIT ON TOOL USE, TASK DECOMPOSITION AND TASK COMPLETETION AND ALLOCATION
    # TODO: Add RLHF Data collection, ask user how the swarm is performing
    # TODO: Create an onboarding process if not settings are preconfigured like `from swarms import Swarm, Swarm()` => then initiate onboarding name your swarm + provide purpose + etc 

    def __init__(self, agents, vectorstore, tools):
        self.agents = agents
        self.vectorstore = vectorstore
        self.tools = tools

    @abstractmethod
    def communicate(self):
        pass

    @abstractmethod
    def run(self):
        pass