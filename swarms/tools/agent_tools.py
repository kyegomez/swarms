

# ---------- Dependencies ----------
import os
import asyncio
import faiss
from typing import Any, Optional, List
from contextlib import contextmanager

from pydantic import BaseModel, Field
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.base import Chain

from langchain.experimental import BabyAGI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import FAISS

from langchain.docstore import InMemoryDocstore
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.tools.file_management.read import ReadFileTool

from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.human.tool import HumanInputRun
from swarms.tools import Terminal, CodeWriter, CodeEditor, process_csv, WebpageQATool

from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool

# ---------- Constants ----------
ROOT_DIR = "./data/"

# ---------- Tools ----------
openai_api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model_name="gpt-4", temperature=1.0, openai_api_key=openai_api_key)

worker_tools = [
    DuckDuckGoSearchRun(),
    WriteFileTool(root_dir=ROOT_DIR),
    ReadFileTool(root_dir=ROOT_DIR),
    process_csv,
    
    WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)),

    # Tool(name='terminal', func=Terminal.execute, description='Operates a terminal'),
    # Tool(name='code_writer', func=CodeWriter(), description='Writes code'),
    # Tool(name='code_editor', func=CodeEditor(), description='Edits code'),#
]

# ---------- Vector Store ----------
embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})