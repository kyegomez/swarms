from swarms.tools.agent_tools import *
from langchain.tools import BaseTool
from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import List, Any, Dict, Optional
from langchain.memory.chat_message_histories import FileChatMessageHistory

import logging
from pydantic import BaseModel, Extra
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkerNodeArgs(BaseModel):
    prompt: str
    run_manager: Optional[CallbackManagerForToolRun] = None

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid

@tool("WorkerNode")
class WorkerNode(BaseTool):
    """Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on"""
    
    args_schema: Optional[Type[BaseModel]] = WorkerNodeArgs
    """Pydantic model class to validate and parse the tool's input arguments."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm = kwargs.get('llm')
        self.tools = kwargs.get('tools')
        self.vectorstore = kwargs.get('vectorstore')
        self.agent = None

    def create_agent(self, ai_name, ai_role, human_in_the_loop, search_kwargs):
        logging.info("Creating agent in WorkerNode")
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=self.tools,
            llm=self.llm,
            memory=self.vectorstore.as_retriever(search_kwargs=search_kwargs),
            human_in_the_loop=human_in_the_loop,
            chat_history_memory=FileChatMessageHistory("chat_history.txt"),
        )
        self.agent.chain.verbose = True

    def _run(
        self, tool_input: Dict[str, Any], run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        prompt = self._parse_input(tool_input)
        tree_of_thoughts_prompt = """
        Imagine three different experts are answering this question. All experts will write down each chain of thought of each step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is...
        """
        self.agent.run([f"{tree_of_thoughts_prompt}{prompt}"])
        return "Task completed by WorkerNode"

    async def _arun(
        self, prompt: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("WorkerNode does not support async")




# class WorkerNode(BaseTool):
#     """Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on """
#     name = "WorkerNode"
#     description = "A worker node that can perform complex tasks"
    
#     def __init__(self, llm, tools, vectorstore):
#         super().__init__()
#         self.llm = llm
#         self.tools = tools
#         self.vectorstore = vectorstore

#     def create_agent(self, ai_name, ai_role, human_in_the_loop, search_kwargs):
#         self.agent = AutoGPT.from_llm_and_tools(
#             ai_name=ai_name,
#             ai_role=ai_role,
#             tools=self.tools,
#             llm=self.llm,
#             memory=self.vectorstore.as_retriever(search_kwargs=search_kwargs),
#             human_in_the_loop=human_in_the_loop,
#         )
#         self.agent.chain.verbose = True

#     # @tool
#         # name="Worker AutoBot Agent",
#         # description="Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on",
#     def _run(
#         self, prompt: str, run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool."""
#         tree_of_thoughts_prompt = """
#         Imagine three different experts are answering this question. All experts will write down each chain of thought of each step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is...
#         """
#         self.agent.run([f"{tree_of_thoughts_prompt}{prompt}"])
#         return "Task completed by WorkerNode"

#     async def _arun(
#         self, prompt: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError("WorkerNode does not support async")