import concurrent.futures
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT

from swarms.tools.autogpt import (
    ReadFileTool,
    VQAinference,
    WriteFileTool,
    compile,
    process_csv,
    query_website_tool,
)
from swarms.utils.decorators import error_decorator, log_decorator, timing_decorator

ROOT_DIR = "./data/"


class Worker:
    """
    Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, 
    it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on

    Parameters:
    - `model_name` (str): The name of the language model to be used (default: "gpt-4").
    - `openai_api_key` (str): The OpenAI API key (optional).
    - `ai_name` (str): The name of the AI worker.
    - `ai_role` (str): The role of the AI worker.
    - `external_tools` (list): List of external tools (optional).
    - `human_in_the_loop` (bool): Enable human-in-the-loop interaction (default: False).
    - `temperature` (float): The temperature parameter for response generation (default: 0.5).
    - `llm` (ChatOpenAI): Pre-initialized ChatOpenAI model instance (optional).
    - `openai` (bool): If True, use the OpenAI language model; otherwise, use `llm` (default: True).
    
    #Usage 
    ```
    from swarms import Worker

    node = Worker(
        ai_name="Optimus Prime",

    )

    task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
    response = node.run(task)
    print(response)
    ```

    """
    @log_decorator
    @error_decorator
    @timing_decorator
    def __init__(
        self, 
        model_name="gpt-4", 
        openai_api_key=None,
        ai_name="Autobot Swarm Worker",
        ai_role="Worker in a swarm",
        external_tools = None,
        human_in_the_loop=False,
        temperature=0.5,
        llm=None,
        openai: bool = True,
    ):
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        self.human_in_the_loop = human_in_the_loop
        self.openai = openai

        
        # if self.openai is True:
        #     try:
        self.llm = ChatOpenAI(
            model_name=model_name, 
            openai_api_key=self.openai_api_key, 
            temperature=self.temperature
        )
            # except Exception as error:
            #     raise RuntimeError(f"Error Initializing ChatOpenAI: {error}")    
        # else:
        #     self.llm = llm(
        #         model_name=model_name, 
        #         temperature=self.temperature
        #     )
            
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.setup_tools(external_tools)
        self.setup_memory()
        self.setup_agent()

        self.task_queue = []
        self.executor = concurrent.futures.ThreadPoolExecutor()
        

    
    def reset(self):
        """
        Reset the message history.
        """
        self.message_history = ["Here is the conversation so far"]
    
    def receieve(self, name: str, message: str) -> None:
        """
        Receive a message and update the message history.

        Parameters:
        - `name` (str): The name of the sender.
        - `message` (str): The received message.
        """
        self.message_history.append(f"{name}: {message}")

    def add(self, task, priority=0):
        self.task_queue.append((priority, task))
    
    def process_task(self, task):
        try:
            result = self.agent.run([task])
            return result
        except Exception as error:
            error_message = f"Error while running task: {str(error)}"
            return error_message
        
    def process_tasks_parallel(self):
        futures = [
            self.executor.submit(
                self.process_task,
                task
            ) for _, task in self.task_queue
        ]
        concurrent.futures.wait(futures)
        results = [future.result() for future in futures]
        return results

    
    @log_decorator
    @error_decorator
    @timing_decorator
    def setup_tools(self, external_tools):
        """
        Set up tools for the worker.

        Parameters:
        - `external_tools` (list): List of external tools (optional).

        Example:
        ```
        external_tools = [MyTool1(), MyTool2()]
        worker = Worker(model_name="gpt-4", 
                openai_api_key="my_key", 
                ai_name="My Worker", 
                ai_role="Worker", 
                external_tools=external_tools, 
                human_in_the_loop=False, 
                temperature=0.5)
        ```
        """
        self.tools = [
            WriteFileTool(root_dir=ROOT_DIR),
            ReadFileTool(root_dir=ROOT_DIR),
            process_csv,
            query_website_tool,
            HumanInputRun(),
            compile,
            VQAinference
        ]
        if external_tools is not None:
            self.tools.extend(external_tools)

    def setup_memory(self):
        """
        Set up memory for the worker.
        """
        try:
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            embedding_size = 4096
            index = faiss.IndexFlatL2(embedding_size)

            self.vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        except Exception as error:
            raise RuntimeError(f"Error setting up memory perhaps try try tuning the embedding size: {error}")
        
    
    def setup_agent(self):
        """
        Set up the autonomous agent.
        """
        try: 
            self.agent = AutoGPT.from_llm_and_tools(
                ai_name=self.ai_name,
                ai_role=self.ai_role,
                tools=self.tools,
                llm=self.llm,
                memory=self.vectorstore.as_retriever(search_kwargs={"k": 8}),
                human_in_the_loop=self.human_in_the_loop
            )
        
        except Exception as error:
            raise RuntimeError(f"Error setting up agent: {error}")
    
    @log_decorator
    @error_decorator
    @timing_decorator
    def run(self, task):
        """
        Run the autonomous agent on a given task.

        Parameters:
        - `task`: The task to be processed.

        Returns:
        - `result`: The result of the agent's processing.
        """
        try:
            result = self.agent.run([task])
            return result
        except Exception as error:
            raise RuntimeError(f"Error while running agent: {error}")
    
    @log_decorator
    @error_decorator
    @timing_decorator
    def __call__(self, task):
        """
        Make the worker callable to run the agent on a given task.

        Parameters:
        - `task`: The task to be processed.

        Returns:
        - `results`: The results of the agent's processing.
        """
        try:
            results = self.agent.run([task])
            return results
        except Exception as error:
            raise RuntimeError(f"Error while running agent: {error}")
