import logging
import asyncio

# from swarms.agents.tools.agent_tools import *
from swarms.agents.tools.agent_tools import *
from swarms.workers.WorkerNode import WorkerNodeInitializer, worker_node
from swarms.boss.boss_node import BossNodeInitializer as BossNode
from swarms.workers.worker_ultra_node import WorkerUltra

from langchain import LLMMathChain


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from swarms.utils.task import Task

# TODO: Pass in abstract LLM class that can utilize Hf or Anthropic models
# TODO: Move away from OPENAI
class Swarms:
    def __init__(self, openai_api_key="", use_vectorstore=True, use_async=True, human_in_the_loop=True):
        #openai_api_key: the openai key. Default is empty
        if not openai_api_key:
            logging.error("OpenAI key is not provided")
            raise ValueError("OpenAI API key is required")
        
        self.openai_api_key = openai_api_key
        self.use_vectorstore = use_vectorstore
        self.use_async = use_async
        self.human_in_the_loop = human_in_the_loop

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
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)

            return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        except Exception as e:
            logging.error(f"Failed to initialize vector store: {e}")
            return None

    def initialize_worker_node(self, worker_tools, vectorstore, llm_class=ChatOpenAI, ai_name="Swarm Worker AI Assistant", human_in_the_loop=True):
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

            worker_node_tool = Tool(name="WorkerNode AI Agent", func=worker_node.run, description="Input: an objective with a todo list for that objective. Output: your task completed: Please be very clear what the objective and task instructions are. The Swarm worker agent is Useful for when you need to spawn an autonomous agent instance as a worker to accomplish any complex tasks, it can search the internet or write code or spawn child multi-modality models to process and generate images and text or audio and so on")
            return worker_node_tool
        except Exception as e:
            logging.error(f"Failed to initialize worker node: {e}")
            raise

    def initialize_boss_node(self, vectorstore, worker_node, llm_class=OpenAI, max_iterations=5, verbose=False):
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
            todo_prompt = PromptTemplate.from_template("You are a boss planer in a swarm who is an expert at coming up with a todo list for a given objective and then creating an worker to help you accomplish your task. Rate every task on the importance of it's probability to complete the main objective on a scale from 0 to 1, an integer. Come up with a todo list for this objective: {objective} and then spawn a worker agent to complete the task for you. Always spawn an worker agent after creating a plan and pass the objective and plan to the worker agent.")
            todo_chain = LLMChain(llm=llm, prompt=todo_prompt)

            #math tool
            llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

            tools = [
                Tool(name="TODO", func=todo_chain.run, description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for your objective. Note create a todo list then assign a ranking from 0.0 to 1.0 to each task, then sort the tasks based on the tasks most likely to achieve the objective. The Output: a todo list for that objective with rankings for each step from 0.1 Please be very clear what the objective is!"),
                worker_node,
                Tool(name="Calculator", func=llm_math_chain.run, description="useful for when you need to answer questions about math")
            ]

            suffix = """Question: {task}\n{agent_scratchpad}"""
            prefix = """You are an Boss in a swarm who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}.\n """
            
            prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["objective", "task", "context", "agent_scratchpad"],)
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=[tool.name for tool in tools])

            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)
            return BossNode(llm, vectorstore, agent_executor, max_iterations=max_iterations)
        except Exception as e:
            logging.error(f"Failed to initialize boss node: {e}")
            raise




    def run_swarms(self, objective):
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
            logging.error(f"An error occurred in run_swarms: {e}")
            return None
        
# usage-# usage-
def swarm(api_key="", objective=""):
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
        swarms = Swarms(api_key, use_async=False) # Turn off async
        result = swarms.run_swarms(objective)
        if result is None:
            logging.error("Failed to run swarms")
        else:
            logging.info(f"Successfully ran swarms with results: {result}")
        return result
    except Exception as e:
        logging.error(f"An error occured in swarm: {e}")
        return None


