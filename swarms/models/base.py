from abc import ABC, abstractmethod


class AbstractModel(ABC):
    # abstract base class for language models
    def __init__():
        pass

    @abstractmethod
    def run(self, prompt):
        # generate text using language model
        pass

    def chat(self, prompt, history):
        pass
