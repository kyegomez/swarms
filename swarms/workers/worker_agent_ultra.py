import logging
import os
from typing import Dict, List

from langchain.memory.chat_message_histories import FileChatMessageHistory

from swarms.agents.tools.agent_tools import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




from typing import Dict, List

from langchain.memory.chat_message_histories import FileChatMessageHistory

from swarms.agents.tools.main import (
    BaseToolSet,
    CodeEditor,
    ExitConversation,
    RequestsGet,
    Terminal,
)
from swarms.utils.main import BaseHandler, CsvToDataframe, FileType


class WorkerUltraNode:
    """Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on"""

    def __init__(self, llm, toolsets, vectorstore):
        if not llm or not toolsets or not vectorstore:
            logging.error("llm, toolsets, and vectorstore cannot be None.")
            raise ValueError("llm, toolsets, and vectorstore cannot be None.")
        
        self.llm = llm
        self.toolsets = toolsets
        self.vectorstore = vectorstore
        self.agent = None

    def create_agent(self, ai_name="Swarm Worker AI Assistant", ai_role="Assistant", human_in_the_loop=False, search_kwargs={}, verbose=False):
        logging.info("Creating agent in WorkerNode")
        try:
            tools_list = list(self.toolsets.values())
            self.agent = AutoGPT.from_llm_and_tools(
                ai_name=ai_name,
                ai_role=ai_role,
                tools=tools_list,  # Pass the dictionary instead of the list
                llm=self.llm,
                memory=self.vectorstore.as_retriever(search_kwargs=search_kwargs),
                human_in_the_loop=human_in_the_loop,
                chat_history_memory=FileChatMessageHistory("chat_history.txt"),
            )
            self.agent.chain.verbose = verbose
        except Exception as e:
            logging.error(f"Error while creating agent: {str(e)}")
            raise e

    def add_toolset(self, toolset: BaseToolSet):
        if not isinstance(toolset, BaseToolSet):
            logging.error("Toolset must be an instance of BaseToolSet.")
            raise TypeError("Toolset must be an instance of BaseToolSet.")
        
        self.toolsets.append(toolset)

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

class WorkerUltraNodeInitializer:
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

    def initialize_toolsets(self):
        try:
            toolsets: List[BaseToolSet] = [
                Terminal(),
                CodeEditor(),
                RequestsGet(),
                ExitConversation(),
            ]
            handlers: Dict[FileType, BaseHandler] = {FileType.DATAFRAME: CsvToDataframe()}

            if os.environ.get("USE_GPU", False):
                import torch

                from swarms.agents.tools.main import (
                    ImageCaptioning,
                    ImageEditing,
                    InstructPix2Pix,
                    Text2Image,
                    VisualQuestionAnswering,
                )

                if torch.cuda.is_available():
                    toolsets.extend(
                        [
                            Text2Image("cuda"),
                            ImageEditing("cuda"),
                            InstructPix2Pix("cuda"),
                            VisualQuestionAnswering("cuda"),
                        ]
                    )
                    handlers[FileType.IMAGE] = ImageCaptioning("cuda")

            return toolsets
        except Exception as e:
            logging.error(f"Failed to initialize toolsets: {e}")

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
            worker_toolsets = self.initialize_toolsets()
            vectorstore = self.initialize_vectorstore()
            worker_node = WorkerUltraNode(llm=self.initialize_llm(llm_class), toolsets=worker_toolsets, vectorstore=vectorstore)
            worker_node.create_agent(ai_name=ai_name, ai_role=ai_role, human_in_the_loop=human_in_the_loop, search_kwargs=search_kwargs, verbose=verbose)
            return worker_node
        except Exception as e:
            logging.error(f"Failed to create worker node: {e}")
            raise

def worker_ultra_node(openai_api_key):
    if not openai_api_key:
        logging.error("OpenAI API key is not provided")
        raise ValueError("OpenAI API key is required")
    
    try:
        initializer = WorkerUltraNodeInitializer(openai_api_key)
        worker_node = initializer.create_worker_node()
        return worker_node
    except Exception as e:
        logging.error(f"An error occurred in worker_node: {e}")
        raise