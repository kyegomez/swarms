from dataclasses import dataclass
from typing import Dict, List

from swarms.tools.tool import BaseTool


@dataclass
class Step:
    """
    Represents a step in a process.

    Attributes:
        task (str): The task associated with the step.
        id (int): The unique identifier of the step.
        dep (List[int]): The list of step IDs that this step depends on.
        args (Dict[str, str]): The arguments associated with the step.
        tool (BaseTool): The tool used to execute the step.
    """

    task: str
    id: int
    dep: List[int]
    args: Dict[str, str]
    tool: BaseTool
