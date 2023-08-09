import logging
from typing import List, Optional, Union

import faiss
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.vectorstores import FAISS
from swarms.agents.tools.autogpt import (
    FileChatMessageHistory,
    ReadFileTool,
    WebpageQATool,
    WriteFileTool,
    DuckDuckGoSearchRun,
    load_qa_with_sources_chain,
    process_csv,
    web_search,
)

# Constants
ROOT_DIR = "./data/"

# Logging configurations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WorkerNodeInitializer:
    """Class to initialize and create autonomous agent instances as worker nodes."""
    
    def __init__(self, openai_api_key: str, worker_name: str = "Swarm Worker AI Assistant", **kwargs):
        self.openai_api_key = openai_api_key
        self.llm = kwargs.get('llm', ChatOpenAI())
        self.tools = kwargs.get('tools', [ReadFileTool(), WriteFileTool()])
        self.worker_name = worker_name
        self.worker_role = kwargs.get('worker_role', "Assistant")
        self.human_in_the_loop = kwargs.get('human_in_the_loop', False)
        self.search_kwargs = kwargs.get('search_kwargs', {})
        self.chat_history_file = kwargs.get('chat_history_file', "chat_history.txt")

        self.create_agent()

    def create_agent(self):
        logging.info("Creating agent in WorkerNode")
        vectorstore = self.initialize_vectorstore()
        try:
            self.agent = AutoGPT.from_llm_and_tools(
                ai_name=self.worker_name,
                ai_role=self.worker_role,
                tools=self.tools,
                llm=self.llm,
                memory=vectorstore,
                human_in_the_loop=self.human_in_the_loop,
                chat_history_memory=FileChatMessageHistory(self.chat_history_file),
            )
        except Exception as e:
            logging.error(f"Error while creating agent: {str(e)}")
            raise

    def add_tool(self, tool: Optional[Tool] = None):
        tool = tool or DuckDuckGoSearchRun()
        
        if not isinstance(tool, Tool):
            raise TypeError("Tool must be an instance of Tool.")
        
        self.tools.append(tool)

    def initialize_vectorstore(self):
        embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        embedding_size = 8192
        index = faiss.IndexFlatL2(embedding_size)
        return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    def run(self, prompt) -> str:
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string.")
        
        try:
            self.agent.run([prompt])
            return "Task completed by WorkerNode"
        except Exception as e:
            logging.error(f"Error running the agent: {str(e)}")
            raise

class WorkerNode:
    """Main WorkerNode class to execute and manage tasks."""
    
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        self.openai_api_key = openai_api_key
        self.worker_node_initializer = WorkerNodeInitializer(openai_api_key)
        self.name = "Swarm Worker AI Assistant"
        self.description = "A worker node that executes tasks"

    def create_worker_node(self, **kwargs):
        worker_name = kwargs.get('worker_name', "Swarm Worker AI Assistant")
        llm_class = kwargs.get('llm_class', ChatOpenAI)
        
        if not llm_class:
            raise ValueError("llm_class cannot be None.")
        
        worker_tools = self.initialize_tools(llm_class)
        vectorstore = self.worker_node_initializer.initialize_vectorstore()
        worker_node = WorkerNodeInitializer(openai_api_key=self.openai_api_key, tools=worker_tools, vectorstore=vectorstore, ai_name=worker_name, **kwargs)
        return worker_node

    def initialize_llm(self, llm_class, temperature):
        return llm_class(openai_api_key=self.openai_api_key, temperature=temperature)

    def initialize_tools(self, llm_class):
        llm = self.initialize_llm(llm_class, temperature=1.0)  # default value for temperature
        tools = [
            web_search,
            WriteFileTool(root_dir=ROOT_DIR),
            ReadFileTool(root_dir=ROOT_DIR),
            process_csv,
            WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)),
        ]
        return tools

def worker_node(openai_api_key):
    """Factory function to create a worker node."""
    
    if not openai_api_key:
        raise ValueError("OpenAI API key is required")
    
    node = WorkerNode(openai_api_key)
    return node.create_worker_node()
