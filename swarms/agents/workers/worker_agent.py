from swarms.tools.agent_tools import *
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import tool


# ---------- Worker Node ----------
from pydantic import Field
#
@tool
class WorkerNode():
    """Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on """
    # name = "WorkerNode"
    # description = "Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on."

    def __init__(self, llm, tools, vectorstore):
        super().__init__()
        self.llm = llm
        self.tools = tools
        self.vectorstore = vectorstore

    def create_agent(self, ai_name, ai_role, human_in_the_loop, search_kwargs):
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name=ai_name,
            ai_role=ai_role,
            tools=self.tools,
            llm=self.llm,
            memory=self.vectorstore.as_retriever(search_kwargs=search_kwargs),
            human_in_the_loop=human_in_the_loop,
        )
        self.agent.chain.verbose = True

    def _run(
        self, prompt: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
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