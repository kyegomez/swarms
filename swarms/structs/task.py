from abc import ABC
#from shapeless import shapeless


#@shapeless
class BaseTask(ABC):
    def __init__(
        self,
        id, 
    ):
        self.id = id

    def add(self):
        pass
    
    def schedule(self, time):
        pass

    def parents(self):
        pass

    def children(self):
        pass

    def preprocess(self):
        pass

    def add_parent(self):
        pass
    
    def is_pending(self):
        pass

    def is_finished(self):
        pass

    def is_executing(self):
        pass

    def run(self):
        pass

    def reset(self):
        pass