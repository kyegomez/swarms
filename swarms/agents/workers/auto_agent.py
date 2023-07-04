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
import os
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
from swarms.agents.workers.auto_agent import MultiModalVisualAgent
from swarms.tools.main import Terminal, CodeWriter, CodeEditor, process_csv, WebpageQATool

class MultiModalVisualAgentTool(BaseTool):
    name = "multi_visual_agent"
    description = "Multi-Modal Visual agent tool"

    def __init__(self, agent: MultiModalVisualAgent):
        self.agent = agent
    
    def _run(self, text: str) -> str:
        #run the multi-modal visual agent with the give task
        return self.agent.run_text(text)




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

        multimodal_agent = MultiModalVisualAgent()
        multimodal_agent_tool = MultiModalVisualAgentTool(multimodal_agent)

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

    




# worker_agent = agent_worker
# tree_of_thoughts_prompt = """

# Imagine three different experts are answering this question. All experts will write down each chain of thought of each step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is...


# """


# #Input problem
# input_problem = """


# Input: 2 8 8 14
# Possible next steps:
# 2 + 8 = 10 (left: 8 10 14)
# 8 / 2 = 4 (left: 4 8 14)
# 14 + 2 = 16 (left: 8 8 16)
# 2 * 8 = 16 (left: 8 14 16)
# 8 - 2 = 6 (left: 6 8 14)
# 14 - 8 = 6 (left: 2 6 8)
# 14 /  2 = 7 (left: 7 8 8)
# 14 - 2 = 12 (left: 8 8 12)
# Input: use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation
# Possible next steps:


# """

# agent.run([f"{tree_of_thoughts_prompt} {input_problem}"])