import logging
import os
from typing import Optional

import faiss
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

from langchain_experimental.autonomous_agents import BabyAGI

from langchain.vectorstores import FAISS
from pydantic import ValidationError



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Boss Node ----------
class BossNodeInitializer:
    """
    The BossNode class is responsible for creating and executing tasks using the BabyAGI model.
    It takes a language model (llm), a vectorstore for memory, an agent_executor for task execution, and a maximum number of iterations for the BabyAGI model.
    """
    def __init__(self, llm, vectorstore, agent_executor, max_iterations, human_in_the_loop):
        if not llm or not vectorstore or not agent_executor or not max_iterations:
            logging.error("llm, vectorstore, agent_executor, and max_iterations cannot be None.")
            raise ValueError("llm, vectorstore, agent_executor, and max_iterations cannot be None.")
        self.llm = llm
        self.vectorstore = vectorstore
        self.agent_executor = agent_executor
        self.max_iterations = max_iterations
        self.human_in_the_loop = human_in_the_loop

        try:
            self.baby_agi = BabyAGI.from_llm(
                llm=self.llm,
                vectorstore=self.vectorstore,
                task_execution_chain=self.agent_executor,
                max_iterations=self.max_iterations,
                human_in_the_loop=self.human_in_the_loop
            )
        except ValidationError as e:
            logging.error(f"Validation Error while initializing BabyAGI: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected Error while initializing BabyAGI: {e}")
            raise

    def initialize_vectorstore(self):
        """
        Init vector store
        """
        try:     
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            embedding_size = 8192
            index = faiss.IndexFlatL2(embedding_size)
            return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            return None
        
    def initialize_llm(self, llm_class, temperature=0.5):
        """
        Init LLM 

        Params:
            llm_class(class): The Language model class. Default is OpenAI.
            temperature (float): The Temperature for the language model. Default is 0.5
        """
        try: 
            # Initialize language model
            return llm_class(openai_api_key=self.openai_api_key, temperature=temperature)
        except Exception as e:
            logging.error(f"Failed to initialize language model: {e}")



    def create_task(self, objective):
        """
        Creates a task with the given objective.
        """
        if not objective:
            logging.error("Objective cannot be empty.")
            raise ValueError("Objective cannot be empty.")
        return {"objective": objective}

    def run(self, task):
        """
        Executes a task using the BabyAGI model.
        """
        if not task:
            logging.error("Task cannot be empty.")
            raise ValueError("Task cannot be empty.")
        try:
            self.baby_agi(task)
        except Exception as e:
            logging.error(f"Error while executing task: {e}")
            raise




class BossNode:
    def __init__(self,
                 llm=None,
                 vectorstore=None,
                 agent_executor=None,
                 max_iterations=5,
                 human_in_the_loop=None,
                 objective: Optional[str] = None,
                 boss_system_prompt: Optional[str] = "You are a boss planner in a swarm...",
                 api_key=None,
                 worker_node=None,
                 llm_class=OpenAI,
                 verbose=False,
                 ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.worker_node = worker_node
        self.boss_system_prompt = boss_system_prompt
        self.llm_class = llm_class
        self.max_iterations = max_iterations
        self.verbose = verbose

        if not self.api_key:
            raise ValueError("[MasterBossNode][ValueError][API KEY must be provided either as an argument or as an environment variable API_KEY]")

        # Initialize components if not provided
        self.llm = llm if llm else self._initialize_llm(self.llm_class)
        self.vectorstore = vectorstore if vectorstore else self._initialize_vectorstore()

        # Setting up todo_chain and agent_executor
        todo_prompt = PromptTemplate.from_template(boss_system_prompt)
        todo_chain = LLMChain(llm=self.llm, prompt=todo_prompt)
        tools = [
            Tool(name="Goal Decomposition Tool", func=todo_chain.run, description="Use Case: Decompose ambitious goals into as many explicit and well defined tasks for an AI agent to follow. Rules and Regulations, don't use this tool too often only in the beginning when the user grants you a mission."),
            Tool(name="Swarm Worker Agent", func=self.worker_node, description="Use Case: When you want to delegate and assign the decomposed goal sub tasks to a worker agent in your swarm, Rules and Regulations, Provide a task specification sheet to the worker agent. It can use the browser, process csvs and generate content")
        ]
        suffix = """Question: {task}\n{agent_scratchpad}"""
        prefix = """You are a Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n """
        prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["objective", "task", "context", "agent_scratchpad"],)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=[tools])
        self.agent_executor = agent_executor if agent_executor else AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=self.verbose)

        # Setup BabyAGI
        try:
            self.baby_agi = BabyAGI.from_llm(
                llm=self.llm,
                vectorstore=self.vectorstore,
                task_execution_chain=self.agent_executor,
                max_iterations=self.max_iterations,
                human_in_the_loop=human_in_the_loop
            )
        except ValidationError as e:
            logging.error(f"Validation Error while initializing BabyAGI: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected Error while initializing BabyAGI: {e}")
            raise

        self.task = self._create_task(objective)

    def _initialize_llm(self, llm_class, temperature=0.5):
        try:
            return llm_class(openai_api_key=self.api_key, temperature=temperature)
        except Exception as e:
            logging.error(f"Failed to initialize language model: {e}")
            raise e

    def _initialize_vectorstore(self):
        try:
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.api_key)
            embedding_size = 8192
            index = faiss.IndexFlatL2(embedding_size)
            return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            return None

    def _create_task(self, objective):
        if not objective:
            logging.error("Objective cannot be empty.")
            raise ValueError("Objective cannot be empty.")
        return {"objective": objective}

    def run(self):
        if not self.task:
            logging.error("Task cannot be empty.")
            raise ValueError("Task cannot be empty.")
        try:
            self.baby_agi(self.task)
        except Exception as e:
            logging.error(f"Error while executing task: {e}")
            raise