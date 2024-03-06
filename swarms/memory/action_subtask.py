from pydantic import BaseModel


class ActionSubtaskEntry(BaseModel):
    """Used to store ActionSubtask data to preserve TaskMemory pointers and context in the form of thought and action.

    Attributes:
    thought: CoT thought string from the LLM.
    action: ReAct action JSON string from the LLM.
    answer: tool-generated and memory-processed response from Griptape.
    """

    thought: str
    action: str
    answer: str
