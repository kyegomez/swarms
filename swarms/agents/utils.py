from abc import ABC, abstractmethod


class LeaderAgent(ABC):
    @abstractmethod
    def distribute_task(self, WAs, task):
        pass

    @abstractmethod
    def collect_results(self, WAs):
        pass

    @abstractmethod
    def process_results(self):
        pass


class WorkerAgent(ABC):
    @abstractmethod
    def execute_task(self):
        pass


class CollaborativeAgent(ABC):
    @abstractmethod
    def execute_task(self, task):
        pass

    @abstractmethod
    def collaborate(self):
        pass


class CompetitiveAgent(ABC):
    @abstractmethod
    def execute_task(self, task):
        pass


def evaluate_results(CompAs):
    pass



# Example
class MyWorkerAgent(WorkerAgent):
    def execute_task(self):
        # Insert your implementation here
        pass
