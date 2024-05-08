from typing import Dict, List, Sequence

from swarms.tools.base_tool import BaseTool
from pydantic import BaseModel


class Step(BaseModel):
    """
    Represents a step in a process.

    Attributes:
        task (str): The task associated with the step.
        id (int): The unique identifier of the step.
        dep (List[int]): The list of step IDs that this step depends on.
        args (Dict[str, str]): The arguments associated with the step.
        tool (BaseTool): The tool used to execute the step.
    """

    task: str = None
    id: int = 0
    dep: List[int] = []
    args: Dict[str, str] = {}
    tool: BaseTool = None
    tools: Sequence[BaseTool] = []
    metadata: Dict[str, str] = {}
