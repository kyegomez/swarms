#aug 10
#Vortex is the name of my Duck pet, ILY Vortex
#Kye

import logging
import os
from typing import Any, List, Optional, Union

import faiss
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT

from swarms.agents.tools.autogpt import (
    FileChatMessageHistory,
    ReadFileTool,
    WebpageQATool,
    WriteFileTool,
    load_qa_with_sources_chain,
    process_csv,
    web_search,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = "./data/"

class VortexWorkerAgent:
    """An autonomous agent instance that accomplishes various language tasks like summarization, text generation of any kind, data analysis, websearch and much more"""

    def __init__(self,
                openai_api_key: str,
                llm: Optional[Union[InMemoryDocstore, ChatOpenAI]] = None,
                tools: Optional[Any] = None,
                embedding_size: Optional[int] = 8192,
                worker_name: Optional[str] = "Vortex Worker Agent",
                worker_role: Optional[str] = "Assistant",
                human_in_the_loop: Optional[bool] = False,
                search_kwargs: dict = {},
                verbose: Optional[bool] = False,
                chat_history_file: str = "chat_history.text"):
        if not openai_api_key:
            raise ValueError("openai_api_key cannot be None, try placing in ENV")
        
        self.openai_api_key = openai_api_key
        self.worker_name = worker_name
        self.worker_role = worker_role
        
        self.embedding_size = embedding_size
        self.human_in_the_loop = human_in_the_loop
        self.search_kwargs = search_kwargs
        
        self.verbose = verbose
        self.chat_history_file = chat_history_file
        self.llm = llm or self.init_llm(ChatOpenAI)
        
        self.tools = tools or self.init_tools()
        self.vectorstore = self.init_vectorstore()
        self.agent = self.create_agent()

    def init_llm(self, llm_class, temperature=1.0):
        try:
            return llm_class(openai_api_key=self.openai_api_key, temperature=temperature)
        except Exception:
            logging.error("Failed to init the language model, make sure the llm function matches the llm abstract type")
            raise
    
    def init_tools(self):
        try:
            logging.info("Initializing tools for VortexWorkerAgent")
            tools = [
                web_search,
                WriteFileTool,
                ReadFileTool,
                process_csv,
                WebpageQATool(qa_chain=load_qa_with_sources_chain(self.llm))
            ]
            return tools
        except Exception as error:
            logging.error(f"Failed to initialize tools: {error}")
            raise
    
    def init_vectorstore(self):
        try:
            openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
            embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
            index = faiss.IndexFlatL2(8192)
            return FAISS(embeddings_model, index, InMemoryDocstore({}), {})
        except Exception as error:
            logging.error(f"Failed to initialize vector store: {error}")
            raise

    def create_agent(self):
        logging.info("Creating agent in VortexWorkerAgent")
        try:
            AutoGPT.from_llm_and_tools(
                ai_name=self.worker_name,
                ai_role=self.worker_role,
                tools=self.tools,
                llm=self.llm,
                memory=self.vectorstore,
                human_in_the_loop=self.human_in_the_loop,
                chat_history_memory=FileChatMessageHistory(self.chat_history_file)
            )
        except Exception as error:
            logging.error(f"Failed while creating agent {str(error)}")
            raise error
    
    def add_tool(self, tool: Tool):
        if not isinstance(tool, Tool):
            logging.error("Tools must be an instant of Tool")
            raise TypeError("Tool must be an instance of Tool, try wrapping your tool with the Tool decorator and fill in the requirements")
        self.tools.append(tool)

    def run(self, prompt) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Prompt must be a non empty string")
        try:
            self.agent.run([prompt])
            return "Task completed by VortexWorkerAgent"
        except Exception as error:
            logging.error(f"While running the agent: {str(error)}")
            raise error
        

