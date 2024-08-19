""" Chatbot with RAG Server """

import asyncio
import logging
import os

# import torch
from contextlib import asynccontextmanager
import langchain
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain,
)
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import (
    ChatMessageHistory,
)
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from swarms.prompts.chat_prompt import Message
from swarms.prompts.conversational_RAG import (
    B_INST,
    B_SYS,
    CONDENSE_PROMPT_TEMPLATE,
    DOCUMENT_PROMPT_TEMPLATE,
    E_INST,
    E_SYS,
    QA_PROMPT_TEMPLATE,
)
from swarms.server.responses import LangchainStreamingResponse
from swarms.server.server_models import ChatRequest, Role
from swarms.server.vector_store import VectorStorage

# Explicitly specify the path to the .env file
# Two folders above the current file's directory
dotenv_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(dotenv_path)

hf_token = os.environ.get(
    "HUGGINFACEHUB_API_KEY"
)  # Get the Huggingface API Token
uploads = os.environ.get(
    "UPLOADS"
)  # Directory where user uploads files to be parsed for RAG
model_dir = os.environ.get("MODEL_DIR")

# hugginface.co model (eg. meta-llama/Llama-2-70b-hf)
model_name = os.environ.get("MODEL_NAME")

# Set OpenAI's API key to 'EMPTY' and API base URL to use vLLM's API server
# or set them to OpenAI API key and base URL.
openai_api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
openai_api_base = (
    os.environ.get("OPENAI_API_BASE") or "http://localhost:8000/v1"
)

env_vars = [
    hf_token,
    uploads,
    model_dir,
    model_name,
    openai_api_key,
    openai_api_base,
]
missing_vars = [var for var in env_vars if not var]

if missing_vars:
    print(
        f"Error: The following environment variables are not set: {', '.join(missing_vars)}"
    )
    exit(1)

useMetal = os.environ.get("USE_METAL", "False") == "True"
use_gpu = os.environ.get("USE_GPU", "False") == "True"

print(f"Uploads={uploads}")
print(f"MODEL_DIR={model_dir}")
print(f"MODEL_NAME={model_name}")
print(f"USE_METAL={useMetal}")
print(f"USE_GPU={use_gpu}")
print(f"OPENAI_API_KEY={openai_api_key}")
print(f"OPENAI_API_BASE={openai_api_base}")

# update tiktoken to include the model name (avoids warning message)
tiktoken.model.MODEL_TO_ENCODING.update(
    {
        model_name: "cl100k_base",
    }
)

print("Logging in to huggingface.co...")
login(token=hf_token)  # login to huggingface.co

langchain.debug = True
langchain.verbose = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the vector store in a background task."""
    print(f"Initializing vector store retrievers for {app.title}.")
    asyncio.create_task(vector_store.init_retrievers())
    yield


chatbot = FastAPI(title="Chatbot", lifespan=lifespan)
router = APIRouter()

current_dir = os.path.dirname(__file__)
print("current_dir: " + current_dir)
static_dir = os.path.join(current_dir, "static")
print("static_dir:  " + static_dir)
chatbot.mount(static_dir, StaticFiles(directory=static_dir), name="static")

chatbot.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Create ./uploads folder if it doesn't exist
uploads = uploads or os.path.join(os.getcwd(), "uploads")
if not os.path.exists(uploads):
    os.makedirs(uploads)

# Initialize the vector store
vector_store = VectorStorage(directory=uploads, use_gpu=use_gpu)


async def create_chain(
    messages: list[Message],
    prompt: PromptTemplate = QA_PROMPT_TEMPLATE,
):
    """Creates the RAG Langchain conversational retrieval chain."""
    print("Creating chain ...")

    llm = ChatOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        model=model_name,
        verbose=True,
        streaming=True,
    )

    # if llm is ALlamaCpp:
    #     llm.max_tokens = max_tokens_to_gen
    # elif llm is AGPT4All:
    #     llm.n_predict = max_tokens_to_gen
    # el
    # if llm is AChatOllama:
    #     llm.max_tokens = max_tokens_to_gen
    # if llm is VLLMAsync:
    #     llm.max_tokens = max_tokens_to_gen

    retriever = await vector_store.get_retriever()

    chat_memory = ChatMessageHistory()
    for message in messages:
        if message.role == Role.USER:
            chat_memory.add_user_message(message.content)
        elif message.role == Role.ASSISTANT:
            chat_memory.add_ai_message(message.content)

    memory = ConversationBufferMemory(
        chat_memory=chat_memory,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_PROMPT_TEMPLATE,
        memory=memory,
        verbose=True,
        output_key="answer",
    )

    stuff_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        output_key="answer",
    )

    doc_chain = StuffDocumentsChain(
        llm_chain=stuff_chain,
        document_variable_name="context",
        document_prompt=DOCUMENT_PROMPT_TEMPLATE,
        verbose=True,
        output_key="answer",
        memory=memory,
    )

    return ConversationalRetrievalChain(
        combine_docs_chain=doc_chain,
        memory=memory,
        retriever=retriever,
        question_generator=question_generator,
        return_generated_question=False,
        return_source_documents=True,
        output_key="answer",
        verbose=True,
    )


router = APIRouter()


@router.post(
    "/chat",
    summary="Chatbot",
    description="Chatbot AI Service",
)
async def chat(request: ChatRequest):
    """ Handles chatbot chat POST requests """
    chain = await create_chain(
        messages=request.messages[:-1],
        prompt=PromptTemplate.from_template(
            f"{B_INST}{B_SYS}{request.prompt.strip()}{E_SYS}{E_INST}"
        ),
    )

    json_config = {
        "question": request.messages[-1].content,
        "chat_history": [
            message.content for message in request.messages[:-1]
        ],
        # "callbacks": [
        #     StreamingStdOutCallbackHandler(),
        #     TokenStreamingCallbackHandler(output_key="answer"),
        #     SourceDocumentsStreamingCallbackHandler(),
        # ],
    }
    return LangchainStreamingResponse(
        chain,
        config=json_config,
    )


chatbot.include_router(router, tags=["chat"])


@chatbot.get("/")
def root():
    """Swarms Chatbot API Root"""
    return {"message": "Swarms Chatbot API"}


@chatbot.get("/favicon.ico")
def favicon():
    """ Returns a favicon """
    file_name = "favicon.ico"
    file_path = os.path.join(chatbot.root_path, "static", file_name)
    return FileResponse(
        path=file_path,
        headers={
            "Content-Disposition": "attachment; filename=" + file_name
        },
    )


logging.basicConfig(level=logging.ERROR)


@chatbot.exception_handler(HTTPException)
async def http_exception_handler(r: Request, exc: HTTPException):
    """Log and return exception details in response."""
    logging.error(
        "HTTPException: %s executing request: %s", exc.detail, r.base_url
    )
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}
    )
