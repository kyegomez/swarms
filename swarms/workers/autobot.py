import faiss
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT

from swarms.agents.tools.autogpt import (
    DuckDuckGoSearchRun,
    FileChatMessageHistory,
    ReadFileTool,
    WebpageQATool,
    WriteFileTool,
    load_qa_with_sources_chain,
    process_csv,
    # web_search,
    query_website_tool
)


ROOT_DIR = "./data/"


class AutoBot:
    def __init__(self, 
                 model_name="gpt-4", 
                 openai_api_key=None,
                 ai_name="Autobot Swarm Worker",
                 ai_role="Worker in a swarm",
                #  embedding_size=None,
                #  k=None,
                 temperature=0.5):
        self.openai_api_key = openai_api_key
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=model_name, 
                            openai_api_key=self.openai_api_key, 
                            temperature=self.temperature)

        self.ai_name = ai_name
        self.ai_role = ai_role

        # self.embedding_size = embedding_size
        # # self.k = k

        self.setup_tools()
        self.setup_memory()
        self.setup_agent()

    def setup_tools(self):
        self.tools = [
            WriteFileTool(root_dir=ROOT_DIR),
            ReadFileTool(root_dir=ROOT_DIR),
            process_csv,
            query_website_tool,
        ]

    def setup_memory(self):
        embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        self.vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    
    def setup_agent(self):
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name=self.ai_name,
            ai_role=self.ai_role,
            tools=self.tools,
            llm=self.llm,
            memory=self.vectorstore.as_retriever(search_kwargs={"k": 8}),
        )
    
    def run(self, task):
        result = self.agent.run([task])
        return result
    
