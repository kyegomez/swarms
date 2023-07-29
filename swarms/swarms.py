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


from swarms.agents.tools.main import WebpageQATool, process_csv
from swarms.boss.boss_node import BossNode
from swarms.workers.worker_node import WorkerNode

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
        openai_api_key: Optional[str] = "", 
        use_vectorstore: Optional[bool] = True, 
        embedding_size: Optional[int] = None, 
        use_async: Optional[bool] = True, 
        worker_name: Optional[str] = "Swarm Worker AI Assistant",
        verbose: Optional[bool] = False,
        human_in_the_loop: Optional[bool] = True, 
        boss_prompt: Optional[str] = "You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n",
        worker_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.5,
        max_iterations: Optional[int] = None,
        logging_enabled: Optional[bool] = True):
            
        self.openai_api_key = openai_api_key
        self.use_vectorstore = use_vectorstore
        self.use_async = use_async
        self.human_in_the_loop = human_in_the_loop
        self.embedding_size = embedding_size
        self.boss_prompt = boss_prompt
        self.worker_prompt = worker_prompt
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.logging_enabled = logging_enabled
        self.verbose = verbose

        self.worker_node = WorkerNode(
            openai_api_key=self.openai_api_key,
            use_vectorstore=self.use_vectorstore,
            embedding_size=self.embedding_size,
            worker_name=self.worker_name,
            worker_prompt=self.worker_prompt,
            temperature=self.temperature,
            human_in_the_loop=self.human_in_the_loop,
            verbose=self.verbose
        )

        self.boss_node = BossNode(
            worker_node=self.worker_node,
            max_iterations=self.max_iterations,
            human_in_the_loop=self.human_in_the_loop,
            embedding_size=self.embedding_size
        )

        self.logger = logging.getLogger()
        if not logging_enabled:
            self.logger.disabled = True

    def run(self, objective):
        """
        Run the swarm with the given objective

        Params:
            objective(str): The task
        """
        try:
            task = self.boss_node.create_task(objective)
            logging.info(f"Running task: {task}")
            if self.use_async:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(self.boss_node.run(task))
            else:
                result = self.boss_node.run(task)
            logging.info(f"Completed tasks: {task}")
            return result
        except Exception as e:
            logging.error(f"An error occurred in run: {e}")
            return None
        
# usage-# usage-
def swarm(
    api_key: Optional[str]="", 
    objective: Optional[str]="", 
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
        swarms = HierarchicalSwarm(api_key, use_async=False) #logging_enabled=logging_enabled) # Turn off async
        result = swarms.run(objective)
        if result is None:
            logging.error("Failed to run swarms")
        else:
            logging.info(f"Successfully ran swarms with results: {result}")
        return result
    except Exception as e:
        logging.error(f"An error occured in swarm: {e}")
        return None




# class HierarchicalSwarm:
#     def __init__(
#         self, 
#         openai_api_key: Optional[str] = "", 

#         use_vectorstore: Optional[bool] = True, 
#         embedding_size: Optional[int] = None, 
#         use_async: Optional[bool] = True, 
#         worker_name: Optional[str] = "Swarm Worker AI Assistant",
#         verbose: Optional[bool] = False,

#         human_in_the_loop: Optional[bool] = True, 
#         boss_prompt: Optional[str] = "You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n",

#         worker_prompt: Optional[str] = None,
#         temperature: Optional[float] = 0.5,
#         max_iterations: Optional[int] = None,
#         logging_enabled: Optional[bool] = True):
            
#         self.openai_api_key = openai_api_key
#         self.use_vectorstore = use_vectorstore

#         self.use_async = use_async
#         self.human_in_the_loop = human_in_the_loop

#         self.embedding_size = embedding_size
#         self.boss_prompt = boss_prompt
#         self.worker_prompt = worker_prompt

#         self.temperature = temperature
#         self.max_iterations = max_iterations
#         self.logging_enabled = logging_enabled

#         self.verbose = verbose

#         self.worker_node = WorkerNode(openai_api_key)

#         self.logger = logging.getLogger()
#         if not logging_enabled:
#             self.logger.disabled = True

#     def initialize_worker_node(self, worker_tools, vectorstore, llm_class=ChatOpenAI):
#         try:    
#             worker_node = self.worker_node.create_worker_node(llm_class=llm_class, ai_name=self.worker_name, ai_role="Assistant", human_in_the_loop=self.human_in_the_loop, search_kwargs={}, verbose=self.verbose)
#             worker_description = self.worker_prompt
#             worker_node_tool = Tool(name="WorkerNode AI Agent", func=worker_node.run, description= worker_description or "Input: an objective with a todo list for that objective. Output: your task completed: Please be very clear what the objective and task instructions are. The Swarm worker agent is Useful for when you need to spawn an autonomous agent instance as a worker to accomplish any complex tasks, it can search the internet or write code or spawn child multi-modality models to process and generate images and text or audio and so on")
#             return worker_node_tool
#         except Exception as e:
#             logging.error(f"Failed to initialize worker node: {e}")
#             raise

#     def initialize_boss_node(self, vectorstore, worker_node, llm_class=OpenAI):
#         """
#         Init BossNode

#         Params:
#             vectorstore (object): the vector store object.
#             worker_node (object): the worker node object
#             llm_class (class): the language model class. Default is OpenAI
#             max_iterations(int): The number of max iterations. Default is 5
#             verbose(bool): Debug mode. Default is False
        
#         """
#         try:

#             # Initialize boss node
#             llm = self.worker_node.initialize_llm(llm_class)
            
#             # prompt = self.boss_prompt
#             todo_prompt = PromptTemplate.from_template(self.boss_prompt)
#             todo_chain = LLMChain(llm=llm, prompt=todo_prompt)

#             tools = [
#                 Tool(name="TODO", func=todo_chain.run, description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for your objective. Note create a todo list then assign a ranking from 0.0 to 1.0 to each task, then sort the tasks based on the tasks most likely to achieve the objective. The Output: a todo list for that objective with rankings for each step from 0.1 Please be very clear what the objective is!"),
#                 worker_node,
#             ]

#             suffix = """Question: {task}\n{agent_scratchpad}"""
#             prefix = """You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n """
            
#             prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["objective", "task", "context", "agent_scratchpad"],)
#             llm_chain = LLMChain(llm=llm, prompt=prompt)
#             agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=[tool.name for tool in tools])

#             agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=self.verbose)
#             return BossNode(llm, vectorstore, agent_executor, self.max_iterations)
#         except Exception as e:
#             logging.error(f"Failed to initialize boss node: {e}")
#             raise

#     def run(self, objective):
#         """
#         Run the swarm with the given objective

#         Params:
#             objective(str): The task
#         """
#         try:
#             # Run the swarm with the given objective
#             worker_tools = self.worker_node.initialize_tools(OpenAI)
#             assert worker_tools is not None, "worker_tools is not initialized"

#             vectorstore = self.worker_node.initialize_vectorstore() if self.use_vectorstore else None
#             assert vectorstore is not None, "vectorstore is not initialized"

#             worker_node = self.initialize_worker_node(worker_tools, vectorstore)

#             boss_node = self.initialize_boss_node(vectorstore, worker_node)

#             task = boss_node.create_task(objective)
#             logging.info(f"Running task: {task}")
#             if self.use_async:
#                 loop = asyncio.get_event_loop()
#                 result = loop.run_until_complete(boss_node.run(task))
#             else:
#                 result = boss_node.run(task)
#             logging.info(f"Completed tasks: {task}")
#             return result
#         except Exception as e:
#             logging.error(f"An error occurred in run: {e}")
#             return None
        
# # usage-# usage-
# def swarm(
#     api_key: Optional[str]="", 
#     objective: Optional[str]="", 
#     ):
#     """
#     Run the swarm with the given API key and objective.

#     Parameters:
#     api_key (str): The OpenAI API key. Default is an empty string.
#     objective (str): The objective. Default is an empty string.

#     Returns:
#     The result of the swarm.
#     """

#     if not api_key or not isinstance(api_key, str):
#         logging.error("Invalid OpenAI key")
#         raise ValueError("A valid OpenAI API key is required")
#     if not objective or not isinstance(objective, str):
#         logging.error("Invalid objective")
#         raise ValueError("A valid objective is required")
#     try:
#         swarms = HierarchicalSwarm(api_key, use_async=False) #logging_enabled=logging_enabled) # Turn off async
#         result = swarms.run(objective)
#         if result is None:
#             logging.error("Failed to run swarms")
#         else:
#             logging.info(f"Successfully ran swarms with results: {result}")
#         return result
#     except Exception as e:
#         logging.error(f"An error occured in swarm: {e}")
#         return None