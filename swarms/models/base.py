from abc import ABC, abstractmethod

class AbstractModel(ABC):
    #abstract base class for language models
    @abstractmethod
    def run(self, prompt):
        #generate text using language model
        pass

    def chat(self, prompt, history):
        pass
    