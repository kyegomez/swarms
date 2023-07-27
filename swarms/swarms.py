import asyncio
import logging
from typing import Optional

import faiss
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.vectorstores import FAISS

from swarms.agents.models.hf import HuggingFaceLLM

# from langchain.tools.human.tool import HumanInputRun
from swarms.agents.tools.main import WebpageQATool, process_csv
from swarms.boss.boss_node import BossNodeInitializer as BossNode
from swarms.workers.worker_node import WorkerNodeInitializer

# from langchain import LLMMathChain
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# TODO: Pass in abstract LLM class that can utilize Hf or Anthropic models, Move away from OPENAI
# TODO: ADD Universal Communication Layer, a ocean vectorstore instance
# TODO: BE MORE EXPLICIT ON TOOL USE, TASK DECOMPOSITION AND TASK COMPLETETION AND ALLOCATION
# TODO: Add RLHF Data collection, ask user how the swarm is performing
# TODO: Create an onboarding process if not settings are preconfigured like `from swarms import Swarm, Swarm()` => then initiate onboarding name your swarm + provide purpose + etc 


# ---------- Constants ----------
ROOT_DIR = "./data/"

class HierarchicalSwarm:
    def __init__(
        self, 
        model_id: Optional[str] = None, 
        openai_api_key: Optional[str] = "", 

        use_vectorstore: Optional[bool] = True, 
        embedding_size: Optional[int] = None, 
        use_async: Optional[bool] = True, 

        human_in_the_loop: Optional[bool] = True, 
        model_type: Optional[str] = None, 
        boss_prompt: Optional[str] = None,

        worker_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
        logging_enabled: Optional[bool] = True):
        
        self.model_id = model_id        
        self.openai_api_key = openai_api_key
        self.use_vectorstore = use_vectorstore

        self.use_async = use_async
        self.human_in_the_loop = human_in_the_loop
        self.model_type = model_type

        self.embedding_size = embedding_size
        self.boss_prompt = boss_prompt
        self.worker_prompt = worker_prompt

        self.temperature = temperature
        self.max_iterations = max_iterations
        self.logging_enabled = logging_enabled

        self.logger = logging.getLogger()
        if not logging_enabled:
            self.logger.disabled = True



    def initialize_llm(self, llm_class: str = None):
        """
        Init LLM 

        Params:
            llm_class(class): The Language model class. Default is OpenAI.
            temperature (float): The Temperature for the language model. Default is 0.5
        """
        try: 
            # Initialize language model
            if self.llm_class == 'openai':
                return OpenAI(openai_api_key=self.openai_api_key, temperature=self.temperature)
            elif self.model_type == "huggingface":
                return HuggingFaceLLM(model_id=self.model_id, temperature=self.temperature)
        except Exception as e:
            logging.error(f"Failed to initialize language model: {e}")

    def initialize_tools(self, llm_class, extra_tools=None):
        """
        Init tools
        
        Params:
            llm_class (class): The Language model class. Default is OpenAI

        extra_tools = [CustomTool()]
            worker_tools = swarms.initialize_tools(OpenAI, extra_tools)
        """
        try:
            llm = self.initialize_llm(llm_class)
            # Initialize tools
            web_search = DuckDuckGoSearchRun()
            tools = [
                web_search,
                WriteFileTool(root_dir=ROOT_DIR),
                ReadFileTool(root_dir=ROOT_DIR),

                process_csv,
                WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)),
            ]

            if extra_tools:
                tools.extend(extra_tools)

            assert tools is not None, "tools is not initialized"
            return tools

        except Exception as e:
            logging.error(f"Failed to initialize tools: {e}")
            raise

    def initialize_vectorstore(self):
        """
        Init vector store
        """
        try:     
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            embedding_size = self.embedding_size
            index = faiss.IndexFlatL2(embedding_size)

            return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            return None

    def initialize_worker_node(self, worker_tools, vectorstore, llm_class=ChatOpenAI, ai_name="Swarm Worker AI Assistant",):
        """
        Init WorkerNode

        Params:
            worker_tools (list): The list of worker tools.
            vectorstore (object): The vector store object
            llm_class (class): The Language model class. Default is ChatOpenAI
            ai_name (str): The AI name. Default is "Swarms worker AI assistant"        
        """
        try:    
            # Initialize worker node
            llm = self.initialize_llm(ChatOpenAI)
            worker_node = WorkerNodeInitializer(llm=llm, tools=worker_tools, vectorstore=vectorstore)
            worker_node.create_agent(ai_name=ai_name, ai_role="Assistant", search_kwargs={}, human_in_the_loop=self.human_in_the_loop) # add search kwargs
            worker_description = self.worker_prompt

            worker_node_tool = Tool(name="WorkerNode AI Agent", func=worker_node.run, description= worker_description or "Input: an objective with a todo list for that objective. Output: your task completed: Please be very clear what the objective and task instructions are. The Swarm worker agent is Useful for when you need to spawn an autonomous agent instance as a worker to accomplish any complex tasks, it can search the internet or write code or spawn child multi-modality models to process and generate images and text or audio and so on")
            return worker_node_tool
        except Exception as e:
            logging.error(f"Failed to initialize worker node: {e}")
            raise

    def initialize_boss_node(self, vectorstore, worker_node, llm_class=OpenAI, max_iterations=None, verbose=False):
        """
        Init BossNode

        Params:
            vectorstore (object): the vector store object.
            worker_node (object): the worker node object
            llm_class (class): the language model class. Default is OpenAI
            max_iterations(int): The number of max iterations. Default is 5
            verbose(bool): Debug mode. Default is False
        
        """
        try:

            # Initialize boss node
            llm = self.initialize_llm(llm_class)
            
            # prompt = self.boss_prompt
            todo_prompt = PromptTemplate.from_template({self.boss_prompt} or "You are a boss planer in a swarm who is an expert at coming up with a todo list for a given objective and then creating an worker to help you accomplish your task. Rate every task on the importance of it's probability to complete the main objective on a scale from 0 to 1, an integer. Come up with a todo list for this objective: {objective} and then spawn a worker agent to complete the task for you. Always spawn an worker agent after creating a plan and pass the objective and plan to the worker agent.")
            todo_chain = LLMChain(llm=llm, prompt=todo_prompt)

            tools = [
                Tool(name="TODO", func=todo_chain.run, description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for your objective. Note create a todo list then assign a ranking from 0.0 to 1.0 to each task, then sort the tasks based on the tasks most likely to achieve the objective. The Output: a todo list for that objective with rankings for each step from 0.1 Please be very clear what the objective is!"),
                worker_node,
            ]

            suffix = """Question: {task}\n{agent_scratchpad}"""
            prefix = """You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n """
            
            prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["objective", "task", "context", "agent_scratchpad"],)
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=[tool.name for tool in tools])

            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
            return BossNode(llm, vectorstore, agent_executor, max_iterations=self.max_iterations)
        except Exception as e:
            logging.error(f"Failed to initialize boss node: {e}")
            raise




    def run(self, objective):
        """
        Run the swarm with the given objective

        Params:
            objective(str): The task
        """
        try:
            # Run the swarm with the given objective
            worker_tools = self.initialize_tools(OpenAI)
            assert worker_tools is not None, "worker_tools is not initialized"

            vectorstore = self.initialize_vectorstore() if self.use_vectorstore else None
            worker_node = self.initialize_worker_node(worker_tools, vectorstore)

            boss_node = self.initialize_boss_node(vectorstore, worker_node)

            task = boss_node.create_task(objective)
            logging.info(f"Running task: {task}")
            if self.use_async:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(boss_node.run(task))
            else:
                result = boss_node.run(task)
            logging.info(f"Completed tasks: {task}")
            return result
        except Exception as e:
            logging.error(f"An error occurred in run: {e}")
            return None
        
# usage-# usage-
def swarm(
    api_key: Optional[str]="", 
    objective: Optional[str]="", 
    model_type: Optional[str]="", 
    model_id: Optional[str]=""
    ):
    """
    Run the swarm with the given API key and objective.

    Parameters:
    api_key (str): The OpenAI API key. Default is an empty string.
    objective (str): The objective. Default is an empty string.

    Returns:
    The result of the swarm.
    """

    if not api_key or not isinstance(api_key, str):
        logging.error("Invalid OpenAI key")
        raise ValueError("A valid OpenAI API key is required")
    if not objective or not isinstance(objective, str):
        logging.error("Invalid objective")
        raise ValueError("A valid objective is required")
    try:
        swarms = HierarchicalSwarm(api_key, model_id=model_type, use_async=False, model_type=model_type) #logging_enabled=logging_enabled) # Turn off async
        result = swarms.run(objective)
        if result is None:
            logging.error("Failed to run swarms")
        else:
            logging.info(f"Successfully ran swarms with results: {result}")
        return result
    except Exception as e:
        logging.error(f"An error occured in swarm: {e}")
        return None

