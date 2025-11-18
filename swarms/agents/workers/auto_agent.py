# General 
import os
import pandas as pd
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT

from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document

import asyncio
import nest_asyncio

# Tools

from contextlib import contextmanager
from typing import Optional
from langchain.agents import tool

from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools import BaseTool, DuckDuckGoSearchRun

from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain

# Memory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

from langchain.tools.human.tool import HumanInputRun
# from swarms.agents.workers.auto_agent import 
from swarms.agents.workers.visual_agent import multimodal_agent_tool
from swarms.tools.main import Terminal, CodeWriter, CodeEditor, process_csv, WebpageQATool



class WorkerAgent:
    def __init__(self, objective: str, api_key: str):
        self.objective = objective
        self.api_key = api_key
        self.worker = self.create_agent_worker()

    def create_agent_worker(self):
        os.environ['OPENAI_API_KEY'] = self.api_key

        llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)
        embeddings_model = OpenAIEmbeddings()
        embedding_size = 1536
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

        query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))
        web_search = DuckDuckGoSearchRun()

        tools = [
            web_search,
            WriteFileTool(root_dir="./data"),
            ReadFileTool(root_dir="./data"),

            multimodal_agent_tool,
            process_csv,
            query_website_tool,
            Terminal,


            CodeWriter,
            CodeEditor
        ]

        agent_worker = AutoGPT.from_llm_and_tools(
            ai_name="WorkerX",
            ai_role="Assistant",
            tools=tools,
            llm=llm,
            memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
            human_in_the_loop=True,
        )

        agent_worker.chain.verbose = True

        return agent_worker

        # objective = "Your objective here"
        # api_key = "Your OpenAI API key here"

        # worker_agent = WorkerAgent(objective, api_key)

    
# objective = "Your objective here"


# worker_agent = WorkerAgent(objective)
