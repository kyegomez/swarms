""" Chatbot with RAG Server """
import logging
import os
from urllib.parse import urlparse
from swarms.structs.agent import Agent
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login
from swarms.prompts.chat_prompt import Message, Role
from swarms.prompts.conversational_RAG import QA_PROMPT_TEMPLATE_STR
from playground.demos.chatbot.server.responses import StreamingResponse
from playground.demos.chatbot.server.server_models import ChatRequest
from playground.demos.chatbot.server.vector_storage import RedisVectorStorage
from swarms.models.popular_llms import OpenAIChatLLM

logging.basicConfig(level=logging.ERROR)

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

# model_dir = os.environ.get("MODEL_DIR")

# hugginface.co model (eg. meta-llama/Llama-2-70b-hf)
# model_name = os.environ.get("MODEL_NAME")

# Set OpenAI's API key to 'EMPTY' and API base URL to use vLLM's API server
# or set them to OpenAI API key and base URL.
openai_api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
openai_api_base = (
    os.environ.get("OPENAI_API_BASE") or "http://localhost:8000/v1"
)

env_vars = [
    hf_token,
    uploads,
    openai_api_key,
    openai_api_base,
]
missing_vars = [var for var in env_vars if not var]

if missing_vars:
    print(
        "Error: The following environment variables are not set: "
        + ", ".join(missing_vars)
    )
    exit(1)

useMetal = os.environ.get("USE_METAL", "False") == "True"
use_gpu = os.environ.get("USE_GPU", "False") == "True"

print(f"Uploads={uploads}")
print(f"USE_METAL={useMetal}")
print(f"USE_GPU={use_gpu}")
print(f"OPENAI_API_KEY={openai_api_key}")
print(f"OPENAI_API_BASE={openai_api_base}")
print("Logging in to huggingface.co...")
login(token=hf_token)  # login to huggingface.co


app = FastAPI(title="Chatbot")
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
# Hardcoded for Swarms documention
URL = "https://docs.swarms.world/en/latest/"
vector_store = RedisVectorStorage(use_gpu=use_gpu)
vector_store.crawl(URL)
print("Vector storage initialized.")


async def create_chat(
    messages: list[Message],
    model_name: str,
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

    doc_retrieval_string = ""
    for message in messages:
        if message.role == Role.HUMAN:
            doc_retrieval_string += f"{Role.HUMAN}:  {message.content}\r\n"
        elif message.role == Role.AI:
            doc_retrieval_string += f"{Role.AI}:  {message.content}\r\n"

    docs = vector_store.embed(messages[-1].content)

    sources = [
        urlparse(URL).scheme + "://" + doc["source_url"]
        for doc in docs
    ]

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
        # docs=[doc["content"] for doc in docs],
        # # docs_folder="docs",
        retry_attempts=3,
        # context_length=1000,
        # tool_schema = dict
        context_length=200000,
        # tool_schema=
        # tools
        # agent_ops_on=True,
    )

    # add chat history messages to short term memory
    for message in messages[:-1]:
        if message.role == Role.HUMAN:
            agent.add_message_to_memory(message.content)
        elif message.role == Role.AI:
            agent.add_message_to_memory(message.content)

    # add docs to short term memory
    for data in [doc["content"] for doc in docs]:
        agent.add_message_to_memory(role=Role.HUMAN, content=data)

    async for response in agent.run_async(messages[-1].content):
        res = response
        res += "\n\nSources:\n"
        for source in sources:
            res += source + "\n"
        yield res


@app.post(
    "/chat",
    summary="Chatbot",
    description="Chatbot AI Service",
)
async def chat(request: ChatRequest):
    """ Handles chatbot chat POST requests """
    response = create_chat(
        messages=request.messages,
        prompt=request.prompt.strip(),
        model_name=request.model.id
    )
    return StreamingResponse(content=response)


@app.get("/")
def root():
    """Swarms Chatbot API Root"""
    return {"message": "Swarms Chatbot API"}


@app.exception_handler(HTTPException)
async def http_exception_handler(r: Request, exc: HTTPException):
    """Log and return exception details in response."""
    logging.error(
        "HTTPException: %s executing request: %s", exc.detail, r.base_url
    )
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}
    )
