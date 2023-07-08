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

class WorkerNode:
    """Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on"""

    def __init__(self, llm, tools, vectorstore):
        self.llm = llm
        self.tools = tools
        self.vectorstore = vectorstore
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

    def add_tool(self, tool: Tool):
        """adds a new tool to the agents toolset"""
        self.tools.append(tool)

    def run(self, tool_input: Dict[str, Any]) -> str:
        if not isinstance(tool_input, dict):
            raise TypeError("tool_input must be a dictionary")
        if 'prompt' not in tool_input:
            raise ValueError("tool_input must contain the key 'prompt'")
        
        prompt = tool_input['prompt']
        if prompt is None:
            raise ValueError("Prompt not found in tool_input")
        
        self.agent.run([f"{prompt}"])
        return "Task completed by WorkerNode"

worker_tool = Tool(
    name="WorkerNode AI Agent",
    func=WorkerNode.run,
    description="Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on"
)



class WorkerNodeInitializer:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def initialize_llm(self, llm_class, temperature=0.5):
        return llm_class(openai_api_key=self.openai_api_key, temperature=temperature)

    def initialize_tools(self, llm_class):
        llm = self.initialize_llm(llm_class)
        web_search = DuckDuckGoSearchRun()
        tools = [
            web_search,
            WriteFileTool(root_dir=ROOT_DIR),
            ReadFileTool(root_dir=ROOT_DIR),
            process_csv,
            WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)),
        ]
        return tools

    def initialize_vectorstore(self):
        embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    def create_worker_node(self, llm_class=ChatOpenAI):
        worker_tools = self.initialize_tools(llm_class)
        vectorstore = self.initialize_vectorstore()
        worker_node = WorkerNode(llm=self.initialize_llm(llm_class), tools=worker_tools, vectorstore=vectorstore)
        worker_node.create_agent(ai_name="Swarm Worker AI Assistant", ai_role="Assistant", human_in_the_loop=False, search_kwargs={})
        return worker_node


# usage
def worker_node(api_key):
    initializer = WorkerNodeInitializer(api_key)
    worker = initializer.create_worker_node()
    return worker