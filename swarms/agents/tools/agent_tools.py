

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
from swarms.agents.tools.main import process_csv, WebpageQATool

from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool

# ---------- Constants ----------
ROOT_DIR = "./data/"
