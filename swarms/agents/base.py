from abc import ABC, abstractmethod

class AbstractAgent(ABC):
    #absrtact agent class
    
    @classmethod
    def __init__(
            self,
            ai_name: str = None,
            ai_role: str = None,
            memory = None,
            tools = None,
            llm = None,
            human_in_the_loop=None,
            output_parser = None,
            chat_history_memory=None,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def run(self, goals=None):
        pass

