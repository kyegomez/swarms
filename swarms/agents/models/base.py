from abc import ABC, abstractmethod

class AbstractModel(ABC):
    #abstract base class for language models

    @abstractmethod
    def __init__(self, model_name **kwargs):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt):
        #generate text using language model
        pass
    