#from shapeless import shapeless

#@shapeless
class ErrorArtifact:
    def __init__(
        self,
        value
    ):
        self.value = value
    
    def __add__(self, other):
        return ErrorArtififact(self.value + other.value)
    
    def to_text(self) -> str:
        return self.value
    