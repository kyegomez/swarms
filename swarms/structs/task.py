from shapeless import shapeless


@shapeless
class Task:
    def __init__(
        self,
        id, 
    ):
        self.id = id

    def forward(self):
        pass

