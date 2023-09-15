import logging
import os
from typing import Optional

import faiss
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_experimental.autonomous_agents import BabyAGI
from pydantic import ValidationError



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Boss Node ----------

class Boss:
    """
    The Bose class is responsible for creating and executing tasks using the BabyAGI model.
    It takes a language model (llm), a vectorstore for memory, an agent_executor for task execution, and a maximum number of iterations for the BabyAGI model.
    
    # Setup
    api_key = "YOUR_OPENAI_API_KEY" # Replace with your OpenAI API Key.
    os.environ["OPENAI_API_KEY"] = api_key

    # Objective for the Boss
    objective = "Analyze website user behavior patterns over the past month."

    # Create a Bose instance
    boss = Bose(
        objective=objective, 
        boss_system_prompt="You are the main controller of a data analysis swarm...", 
        api_key=api_key, 
        worker_node=WorkerNode
    )

    # Run the Bose to process the objective
    boss.run()
    """
    def __init__(
            self, 
            objective: str, 
            api_key=None, 
            max_iterations=5, 
            human_in_the_loop=None, 
            boss_system_prompt="You are a boss planner in a swarm...", 
            llm_class=OpenAI, 
            worker_node=None, 
            verbose=False
        ):
        # Store parameters
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.objective = objective
        self.max_iterations = max_iterations
        self.boss_system_prompt = boss_system_prompt
        self.llm_class = llm_class
        self.verbose = verbose
        
        # Initialization methods
        self.llm = self._initialize_llm()
        self.vectorstore = self._initialize_vectorstore()
        self.task = self._create_task(self.objective)
        self.agent_executor = self._initialize_agent_executor(worker_node)
        self.baby_agi = self._initialize_baby_agi(human_in_the_loop)

    def _initialize_llm(self):
        """
        Init LLM 

        Params:
            llm_class(class): The Language model class. Default is OpenAI.
            temperature (float): The Temperature for the language model. Default is 0.5
        """
        try:
            return self.llm_class(openai_api_key=self.api_key, temperature=0.5)
        except Exception as e:
            logging.error(f"Failed to initialize language model: {e}")
            raise e

    def _initialize_vectorstore(self):
        try:
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.api_key)
            embedding_size = 8192
            index = faiss.IndexFlatL2(embedding_size)

            return FAISS(
                embeddings_model.embed_query, 
                index, 
                InMemoryDocstore({}), {}
            )
        
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            raise e

    def _initialize_agent_executor(self, worker_node):
        todo_prompt = PromptTemplate.from_template(self.boss_system_prompt)
        todo_chain = LLMChain(llm=self.llm, prompt=todo_prompt)
        tools = [
            Tool(
                name="Goal Decomposition Tool", 
                func=todo_chain.run, 
                description="Use Case: Decompose ambitious goals into as many explicit and well defined tasks for an AI agent to follow. Rules and Regulations, don't use this tool too often only in the beginning when the user grants you a mission."
            ),
            Tool(name="Swarm Worker Agent", func=worker_node, description="Use Case: When you want to delegate and assign the decomposed goal sub tasks to a worker agent in your swarm, Rules and Regulations, Provide a task specification sheet to the worker agent. It can use the browser, process csvs and generate content")
        ]

        suffix = """Question: {task}\n{agent_scratchpad}"""
        prefix = """You are a Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n """
        prompt = ZeroShotAgent.create_prompt(
            tools, 
            prefix=prefix, 
            suffix=suffix, 
            input_variables=["objective", "task", "context", "agent_scratchpad"],
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tools)
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=self.verbose)

    def _initialize_baby_agi(self, human_in_the_loop):
        try:
            return BabyAGI.from_llm(
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
