""" Chatbot with RAG Server """

import asyncio
import logging
import os

# import torch
from contextlib import asynccontextmanager
from typing import AsyncIterator
from swarms.structs.agent import Agent
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login

from swarms.prompts.chat_prompt import Message, Role
from swarms.prompts.conversational_RAG import (
    B_INST,
    B_SYS,
    CONDENSE_PROMPT_TEMPLATE,
    DOCUMENT_PROMPT_TEMPLATE,
    E_INST,
    E_SYS,
    QA_PROMPT_TEMPLATE_STR,
)
from playground.demos.chatbot.server.responses import StreamingResponse
from playground.demos.chatbot.server.server_models import ChatRequest
from playground.demos.chatbot.server.vector_store import VectorStorage
from swarms.models.popular_llms import OpenAIChatLLM

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the vector store in a background task."""
    asyncio.create_task(vector_store.init_retrievers())
    yield


app = FastAPI(title="Chatbot", lifespan=lifespan)
router = APIRouter()

current_dir = os.path.dirname(__file__)
print("current_dir: " + current_dir)
static_dir = os.path.join(current_dir, "static")
print("static_dir:  " + static_dir)
app.mount(static_dir, StaticFiles(directory=static_dir), name="static")

app.add_middleware(
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


async def create_chat(
    messages: list[Message],
    prompt: str = QA_PROMPT_TEMPLATE_STR,
):
    """Creates the RAG conversational retrieval chain."""
    print("Creating chat from history and relevant docs if any ...")

    llm = OpenAIChatLLM(
        api_key=openai_api_key,
        base_url=openai_api_base,
        model=model_name,
        verbose=True,
        streaming=True,
    )

    retriever = await vector_store.get_retriever("swarms")
    doc_retrieval_string = ""
    for message in messages:
        if message.role == Role.HUMAN:
            doc_retrieval_string += f"{Role.HUMAN}:  {message.content}\r\n"
        elif message.role == Role.AI:
            doc_retrieval_string += f"{Role.AI}:  {message.content}\r\n"

    docs = retriever.invoke(doc_retrieval_string)

    # find {context} in prompt and replace it with the docs page_content.
    # Concatenate the content of all documents
    context = "\n".join(doc.page_content for doc in docs)

    # Replace {context} in the prompt with the concatenated document content
    prompt = prompt.replace("{context}", context)

    # Replace {chat_history} in the prompt with doc_retrieval_string
    prompt = prompt.replace("{chat_history}", doc_retrieval_string)

    # Replace {question} in the prompt with the last message.
    prompt = prompt.replace("{question}", messages[-1].content)

    # Initialize the agent
    agent = Agent(
        agent_name="Swarms QA ChatBot",
        system_prompt=prompt,
        llm=llm,
        max_loops=1,
        autosave=True,
        # dynamic_temperature_enabled=True,
        dashboard=False,
        verbose=True,
        streaming_on=True,
        # interactive=True, # Set to False to disable interactive mode
        dynamic_temperature_enabled=False,
        saved_state_path="chatbot.json",
        # tools=[#Add your functions here# ],
        # stopping_token="Stop!",
        # interactive=True,
        # docs_folder="docs", # Enter your folder name
        # pdf_path="docs/finance_agent.pdf",
        # sop="Calculate the profit for a company.",
        # sop_list=["Calculate the profit for a company."],
        user_name="RAH@EntangleIT.com",
        docs=[doc.page_content for doc in docs],
        # # docs_folder="docs",
        retry_attempts=3,
        # context_length=1000,
        # tool_schema = dict
        context_length=200000,
        # tool_schema=
        # tools
        # agent_ops_on=True,
    )

    for message in messages[:-1]:
        if message.role == Role.HUMAN:
            agent.add_message_to_memory(message.content)
        elif message.role == Role.AI:
            agent.add_message_to_memory(message.content)

    async for response in agent.run_async(messages[-1].content):
        yield response

    # memory = ConversationBufferMemory(
    #     chat_memory=chat_memory,
    #     memory_key="chat_history",
    #     input_key="question",
    #     output_key="answer",
    #     return_messages=True,
    # )

    # question_generator = LLMChain(
    #     llm=llm,
    #     prompt=CONDENSE_PROMPT_TEMPLATE,
    #     memory=memory,
    #     verbose=True,
    #     output_key="answer",
    # )

    # stuff_chain = LLMChain(
    #     llm=llm,
    #     prompt=prompt,
    #     verbose=True,
    #     output_key="answer",
    # )

    # doc_chain = StuffDocumentsChain(
    #     llm_chain=stuff_chain,
    #     document_variable_name="context",
    #     document_prompt=DOCUMENT_PROMPT_TEMPLATE,
    #     verbose=True,
    #     output_key="answer",
    #     memory=memory,
    # )

    # return ConversationalRetrievalChain(
    #     combine_docs_chain=doc_chain,
    #     memory=memory,
    #     retriever=retriever,
    #     question_generator=question_generator,
    #     return_generated_question=False,
    #     return_source_documents=True,
    #     output_key="answer",
    #     verbose=True,
    # )

@app.post(
    "/chat",
    summary="Chatbot",
    description="Chatbot AI Service",
)
async def chat(request: ChatRequest):
    """ Handles chatbot chat POST requests """
    response = create_chat(
        messages=request.messages,
        prompt=request.prompt.strip()
    )
    # return response
    return StreamingResponse(content=response)

    # json_config = {
    #     "question": request.messages[-1].content,
    #     "chat_history": [
    #         message.content for message in request.messages[:-1]
    #     ],
    #     # "callbacks": [
    #     #     StreamingStdOutCallbackHandler(),
    #     #     TokenStreamingCallbackHandler(output_key="answer"),
    #     #     SourceDocumentsStreamingCallbackHandler(),
    #     # ],
    # }
    # return LangchainStreamingResponse(
    #     chain=chain,
    #     config=json_config,
    #     run_mode="async"
    # )

@app.get("/")
def root():
    """Swarms Chatbot API Root"""
    return {"message": "Swarms Chatbot API"}


@app.get("/favicon.ico")
def favicon():
    """ Returns a favicon """
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "static", file_name)
    return FileResponse(
        path=file_path,
        headers={
            "Content-Disposition": "attachment; filename=" + file_name
        },
    )


logging.basicConfig(level=logging.ERROR)


@app.exception_handler(HTTPException)
async def http_exception_handler(r: Request, exc: HTTPException):
    """Log and return exception details in response."""
    logging.error(
        "HTTPException: %s executing request: %s", exc.detail, r.base_url
    )
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}
    )
