import logging

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

    def __init__(self, llm, tools, vectorstore):
        if not llm or not tools or not vectorstore:
            logging.error("llm, tools, and vectorstore cannot be None.")
            raise ValueError("llm, tools, and vectorstore cannot be None.")
        
        self.llm = llm
        self.tools = tools
        self.vectorstore = vectorstore
        self.agent = None

    def create_agent(self, ai_name="Swarm Worker AI Assistant", ai_role="Assistant", human_in_the_loop=False, search_kwargs={}, verbose=False):
        logging.info("Creating agent in WorkerNode")
        try:
            self.agent = AutoGPT.from_llm_and_tools(
                ai_name=ai_name,
                ai_role=ai_role,
                tools=self.tools,
                llm=self.llm,
                memory=self.vectorstore.as_retriever(search_kwargs=search_kwargs),
                human_in_the_loop=human_in_the_loop,
                chat_history_memory=FileChatMessageHistory("chat_history.txt"),
            )
            # self.agent.chain.verbose = verbose
        except Exception as e:
            logging.error(f"Error while creating agent: {str(e)}")
            raise e


    def add_tool(self, tool: Tool):
        if not isinstance(tool, Tool):
            logging.error("Tool must be an instance of Tool.")
            raise TypeError("Tool must be an instance of Tool.")
        
        self.tools.append(tool)

    def run(self, prompt: str) -> str:
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

    def initialize_vectorstore(self):
        try:
                
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)
            return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            raise

    def create_worker_node(self, llm_class=ChatOpenAI, ai_name="Swarm Worker AI Assistant", ai_role="Assistant", human_in_the_loop=False, search_kwargs={}, verbose=False):
        if not llm_class:
            logging.error("llm_class cannot be None.")
            raise ValueError("llm_class cannot be None.")
        try:
            worker_tools = self.initialize_tools(llm_class)
            vectorstore = self.initialize_vectorstore()
            worker_node = WorkerNode(llm=self.initialize_llm(llm_class), tools=worker_tools, vectorstore=vectorstore)
            worker_node.create_agent(ai_name=ai_name, ai_role=ai_role, human_in_the_loop=human_in_the_loop, search_kwargs=search_kwargs, verbose=verbose)
            return worker_node
        except Exception as e:
            logging.error(f"Failed to create worker node: {e}")
            raise

def worker_node(openai_api_key):
    if not openai_api_key:
        logging.error("OpenAI API key is not provided")
        raise ValueError("OpenAI API key is required")
    
    try:

        initializer = WorkerNodeInitializer(openai_api_key)
        worker_node = initializer.create_worker_node()
        return worker_node
    except Exception as e:
        logging.error(f"An error occured in worker_node: {e}")
        raise


