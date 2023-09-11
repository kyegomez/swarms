import faiss
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT

from swarms.tools.autogpt import (
    ReadFileTool,
    WriteFileTool,
    process_csv,
    # web_search,
    query_website_tool,
)
from swarms.utils.decorators import error_decorator, log_decorator, timing_decorator

ROOT_DIR = "./data/"


class Worker:
    """Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks, it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on"""
    @log_decorator
    @error_decorator
    @timing_decorator
    def __init__(self, 
                 model_name="gpt-4", 
                 openai_api_key=None,
                 ai_name="Autobot Swarm Worker",
                 ai_role="Worker in a swarm",
                 external_tools = None,
                human_in_the_loop=False,
                 temperature=0.5):
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        self.human_in_the_loop = human_in_the_loop

        
        try:
            self.llm = ChatOpenAI(model_name=model_name, 
                                openai_api_key=self.openai_api_key, 
                                temperature=self.temperature)
        except Exception as error:
            raise RuntimeError(f"Error Initializing ChatOpenAI: {error}")    
        
        self.ai_name = ai_name
        self.ai_role = ai_role

        # self.embedding_size = embedding_size
        # # self.k = k

        self.setup_tools(external_tools)
        self.setup_memory()
        self.setup_agent()
    
    @log_decorator
    @error_decorator
    @timing_decorator
    def setup_tools(self, external_tools):
        """
        external_tools = [MyTool1(), MyTool2()]
        worker = Worker(model_name="gpt-4", 
                openai_api_key="my_key", 
                ai_name="My Worker", 
                ai_role="Worker", 
                external_tools=external_tools, 
                human_in_the_loop=False, 
                temperature=0.5)
        
        """
        self.tools = [
            WriteFileTool(root_dir=ROOT_DIR),
            ReadFileTool(root_dir=ROOT_DIR),
            process_csv,
            query_website_tool,
            HumanInputRun(),
            #zapier
            #email
            #pdf
            # Tool(name="Goal Decomposition Tool", func=todo_chain.run, description="Use Case: Decompose ambitious goals into as many explicit and well defined tasks for an AI agent to follow. Rules and Regulations, don't use this tool too often only in the beginning when the user grants you a mission."),
        ]
        if external_tools is not None:
            self.tools.extend(external_tools)

    def setup_memory(self):
        try:
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)
            self.vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        except Exception as error:
            raise RuntimeError(f"Error setting up memory perhaps try try tuning the embedding size: {error}")
        
    
    def setup_agent(self):
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
        try:
            result = self.agent.run([task])
            return result
        except Exception as error:
            raise RuntimeError(f"Error while running agent: {error}")
    
    @log_decorator
    @error_decorator
    @timing_decorator
    def __call__(self, task):
        try:
            results = self.agent.run([task])
            return results
        except Exception as error:
            raise RuntimeError(f"Error while running agent: {error}")