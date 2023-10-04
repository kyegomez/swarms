from typing import Dict, List

from langchain.base_language import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain_experimental.autonomous_agents.hugginggpt.repsonse_generator import (
    load_response_generator,
)
from langchain_experimental.autonomous_agents.hugginggpt.task_executor import (
    TaskExecutor,
)
from langchain_experimental.autonomous_agents.hugginggpt.task_planner import (
    load_chat_planner,
)
from transformers import load_tool

class Step:
    def __init__(
        self,
        task: str,
        id: int,
        dep: List[int],
        args: Dict[str, str],
        tool: BaseTool
    ):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool

class Plan:
    def __init__(
        self,
        steps: List[Step]
    ):
        self.steps = steps
    
    def __str__(self) -> str:
        return str([str(step) for step in self.steps])
    
    def __repr(self) -> str:
        return str(self)





class OmniModalAgent:
    """
    OmniModalAgent
    LLM -> Plans -> Tasks -> Tools -> Response

    Architecture:
    1. LLM: Language Model
    2. Chat Planner: Plans
    3. Task Executor: Tasks
    4. Tools: Tools

    Args:
        llm (BaseLanguageModel): Language Model
        tools (List[BaseTool]): List of tools

    Returns:
        str: response

    Usage:
    from swarms import OmniModalAgent, OpenAIChat,

    llm = OpenAIChat()
    agent = OmniModalAgent(llm)
    response = agent.run("Hello, how are you? Create an image of how your are doing!")
    """
    def __init__(
        self,
        llm: BaseLanguageModel,
        # tools: List[BaseTool]
    ):
        self.llm = llm
        
        print("Loading tools...")
        self.tools = [
            load_tool(tool_name)
            for tool_name in [
                "document-question-answering",
                "image-captioning",
                "image-question-answering",
                "image-segmentation",
                "speech-to-text",
                "summarization",
                "text-classification",
                "text-question-answering",
                "translation",
                "huggingface-tools/text-to-image",
                "huggingface-tools/text-to-video",
                "text-to-speech",
                "huggingface-tools/text-download",
                "huggingface-tools/image-transformation",
            ]
        ]
        
        self.chat_planner = load_chat_planner(llm)
        self.response_generator = load_response_generator(llm)
        # self.task_executor = TaskExecutor
    

    def run(self, input: str) -> str:
        """Run the OmniAgent"""
        plan = self.chat_planner.plan(
            inputs={
                "input": input,
                "hf_tools": self.tools,
            }
        )
        self.task_executor = TaskExecutor(plan)
        self.task_executor.run()

        response = self.response_generator.generate(
            {"task_execution": self.task_executor}
        )

        return response


