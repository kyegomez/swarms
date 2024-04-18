import abc
import json
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import List, Optional


@unique
class LLMStatusCode(Enum):
    SUCCESS = 0
    ERROR = 1


@unique
class NodeType(Enum):
    action = auto()
    trigger = auto()


@unique
class WorkflowType(Enum):
    Main = auto()
    Sub = auto()


@unique
class ToolCallStatus(Enum):
    ToolCallSuccess = auto()
    ToolCallPartlySuccess = auto()
    NoSuchTool = auto()
    NoSuchFunction = auto()
    InputCannotParsed = auto()

    UndefinedParam = auto()
    ParamTypeError = auto()
    UnSupportedParam = auto()
    UnsupportedExpression = auto()
    ExpressionError = auto()
    RequiredParamUnprovided = auto()


@unique
class TestDataType(Enum):
    NoInput = auto()
    TriggerInput = auto()
    ActionInput = auto()
    SubWorkflowInput = auto()


@unique
class RunTimeStatus(Enum):
    FunctionExecuteSuccess = auto()
    TriggerAcivatedSuccess = auto()
    ErrorRaisedHere = auto()
    ErrorRaisedInner = auto()
    DidNotImplemented = auto()
    DidNotBeenCalled = auto()


@dataclass
class TestResult:
    """
    Responsible for handling the data structure of [{}]
    """

    data_type: TestDataType = TestDataType.ActionInput

    input_data: Optional[list] = field(default_factory=lambda: [])

    runtime_status: RunTimeStatus = RunTimeStatus.DidNotBeenCalled
    visit_times: int = 0

    error_message: str = ""
    output_data: Optional[list] = field(default_factory=lambda: [])

    def load_from_json(self):
        pass

    def to_json(self):
        pass

    def to_str(self):
        prompt = f"""
This function has been executed for {self.visit_times} times. Last execution:
1.Status: {self.runtime_status.name}
2.Input: 
{self.input_data}

3.Output:
{self.output_data}"""
        return prompt


@dataclass
class Action:
    content: str = ""
    thought: str = ""
    plan: List[str] = field(default_factory=lambda: [])
    criticism: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=lambda: {})

    tool_output_status: ToolCallStatus = ToolCallStatus.ToolCallSuccess
    tool_output: str = ""

    def to_json(self):
        try:
            tool_output = json.loads(self.tool_output)
        except:
            tool_output = self.tool_output
        return {
            "thought": self.thought,
            "plan": self.plan,
            "criticism": self.criticism,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output_status": self.tool_output_status.name,
            "tool_output": tool_output,
        }


@dataclass
class userQuery:
    task: str
    additional_information: List[str] = field(default_factory=lambda: [])
    refine_prompt: str = field(default_factory=lambda: "")

    def print_self(self):
        lines = [self.task]
        for info in self.additional_information:
            lines.append(f"- {info}")
        return "\n".join(lines)


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractSingleton(abc.ABC, metaclass=Singleton):
    """
    Abstract singleton class for ensuring only one instance of a class.
    """
