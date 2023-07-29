import logging
from typing import Optional, List, Union

import faiss
from langchain.agents import Tool
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.vectorstores import FAISS

# from langchain.tools.human.tool import HumanInputRun
from swarms.agents.tools.main import WebpageQATool, process_csv


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



ROOT_DIR = "./data/"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WorkerNodeInitializer:
    """Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on"""

    def __init__(self, 
                 llm: Optional[Union[InMemoryDocstore, ChatOpenAI]] = None, 
                 tools: Optional[List[Tool]] = None, 
                 vectorstore: Optional[FAISS] = None,
                 embedding_size: Optional[int] = 1926,
                 worker_name: Optional[str] = "Swarm Worker AI Assistant", 
                 worker_role: Optional[str] = "Assistant", 
                 human_in_the_loop: Optional[bool] = False, 
                 search_kwargs: dict = {}, 
                 verbose: Optional[bool] = False,
                 chat_history_file: str = "chat_history.txt"):
        
        self.llm = llm if llm is not None else ChatOpenAI()
        self.tools = tools if tools is not None else [ReadFileTool(), WriteFileTool()]
        self.vectorstore = vectorstore

        # Initializing agent in the constructor
        self.worker_name = worker_name
        self.worker_role = worker_role
        self.embedding_size = embedding_size
        self.human_in_the_loop = human_in_the_loop
        self.search_kwargs = search_kwargs
        self.verbose = verbose
        self.chat_history_file = chat_history_file

        self.create_agent()

    def create_agent(self):
        
        logging.info("Creating agent in WorkerNode")
        try:
            # if self.vectorstore is None:
            #     raise ValueError("Vectorstore is not initialized in WorkerNodeInitializer")
            vectorstore = self.initialize_vectorstore()
            
            self.agent = AutoGPT.from_llm_and_tools(
                worker_name=self.worker_name,
                worker_role=self.worker_role,
                tools=self.tools,
                llm=self.llm,
                memory=vectorstore,
                human_in_the_loop=self.human_in_the_loop,
                chat_history_memory=FileChatMessageHistory(self.chat_history_file),
            )
            # self.agent.chain.verbose = verbose
        except Exception as e:
            logging.error(f"Error while creating agent: {str(e)}")
            raise e

    def add_tool(self, tool: Optional[Tool] = None):
        if tool is None:
            tool = DuckDuckGoSearchRun()
        
        if not isinstance(tool, Tool):
            logging.error("Tool must be an instance of Tool.")
            raise TypeError("Tool must be an instance of Tool.")
        
        self.tools.append(tool)

    def initialize_vectorstore(self):
        try:
            embedding_size = self.embedding_size
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            embedding_size = embedding_size
            index = faiss.IndexFlatL2(embedding_size=embedding_size)
            return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            return None

    def run(self, prompt) -> str:
        if not isinstance(prompt, str):
            logging.error("Prompt must be a string.")
            raise TypeError("Prompt must be a string.")
        
        if not prompt:
            logging.error("Prompt is empty.")
            raise ValueError("Prompt is empty.")
        
        try:
            self.agent.run([f"{prompt}"])
            return "Task completed by WorkerNode"
        except Exception as e:
            logging.error(f"While running the agent: {str(e)}")
            raise e

class WorkerNode:
    def __init__(self, openai_api_key):
        if not openai_api_key:
            logging.error("OpenAI API key is not provided")
            raise ValueError("openai_api_key cannot be None")
        
        self.openai_api_key = openai_api_key
        self.worker_node_initializer = WorkerNodeInitializer(openai_api_key)

    def initialize_llm(self, llm_class, temperature=0.5):
        if not llm_class:
            logging.error("llm_class cannot be none")
            raise ValueError("llm_class cannot be None")
        
        try:
            return llm_class(openai_api_key=self.openai_api_key, temperature=temperature)
        except Exception as e:
            logging.error(f"Failed to initialize language model: {e}")
            raise

    def initialize_tools(self, llm_class):
        if not llm_class:
            logging.error("llm_class not cannot be none")
            raise ValueError("llm_class cannot be none")
        try:
            logging.info('Creating WorkerNode')
            llm = self.initialize_llm(llm_class)
            web_search = DuckDuckGoSearchRun()

            tools = [
                web_search,
                WriteFileTool(root_dir=ROOT_DIR),
                ReadFileTool(root_dir=ROOT_DIR),
                process_csv,
                WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)),
            ]
            if not tools:
                logging.error("Tools are not initialized")
                raise ValueError("Tools are not initialized")
            return tools
        except Exception as e:
            logging.error(f"Failed to initialize tools: {e}")

    def create_worker_node(self, llm_class=ChatOpenAI, ai_name="Swarm Worker AI Assistant", ai_role="Assistant", human_in_the_loop=False, search_kwargs={}, verbose=False):
        if not llm_class:
            logging.error("llm_class cannot be None.")
            raise ValueError("llm_class cannot be None.")
        try:
            worker_tools = self.initialize_tools(llm_class)
            vectorstore = self.worker_node_initializer.initialize_vectorstore()
            worker_node = WorkerNodeInitializer(
                openai_api_key=self.openai_api_key,
                llm=self.initialize_llm(llm_class), 
                tools=worker_tools, 
                vectorstore=vectorstore,
                ai_name=ai_name, 
                ai_role=ai_role, 
                human_in_the_loop=human_in_the_loop, 
                search_kwargs=search_kwargs, 
                verbose=verbose
            )
            return worker_node
        except Exception as e:
            logging.error(f"Failed to create worker node: {e}")
            raise

def worker_node(openai_api_key):
    if not openai_api_key:
        logging.error("OpenAI API key is not provided")
        raise ValueError("OpenAI API key is required")
    
    try:
        worker_node = WorkerNode(openai_api_key)
        return worker_node.create_worker_node()
    except Exception as e:
        logging.error(f"An error occured in worker_node: {e}")
        raise