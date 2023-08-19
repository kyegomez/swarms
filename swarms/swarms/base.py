from abc import ABC, abstractmethod

class AbstractSwarm(ABC):

    def __init__(self, agents, vectorstore, tools):
        self.agents = agents
        self.vectorstore = vectorstore
        self.tools = tools

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def communicate(self):
        pass

    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def solve(self):
        pass