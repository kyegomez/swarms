from abc import ABC, abstractmethod

class AbstractModel(ABC):
    #abstract base class for language models
    @abstractmethod
    def generate(self, prompt):
        #generate text using language model
        pass

    def chat(self, prompt, history):
        pass
    