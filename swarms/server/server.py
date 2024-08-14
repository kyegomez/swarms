import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List

import langchain
from pydantic import ValidationError, parse_obj_as
from swarms.prompts.chat_prompt import Message
from swarms.server.callback_handlers import SourceDocumentsStreamingCallbackHandler, TokenStreamingCallbackHandler
import tiktoken

# import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
from huggingface_hub import login
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationStringBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from swarms.server.responses import LangchainStreamingResponse
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from swarms.prompts.conversational_RAG import (
    B_INST,
    B_SYS,
    CONDENSE_PROMPT_TEMPLATE,
    DOCUMENT_PROMPT_TEMPLATE,
    E_INST,
    E_SYS,
    QA_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
)

from swarms.server.vector_store import VectorStorage

from swarms.server.server_models import (
    ChatRequest,
    LogMessage,
    AIModel,
    AIModels,
    RAGFile,
    RAGFiles,
    Role,
    State,
    GetRAGFileStateRequest,
    ProcessRAGFileRequest
)

# Explicitly specify the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

hf_token = os.environ.get("HUGGINFACEHUB_API_KEY") # Get the Huggingface API Token
uploads = os.environ.get("UPLOADS") # Directory where user uploads files to be parsed for RAG
model_dir = os.environ.get("MODEL_DIR")

# hugginface.co model (eg. meta-llama/Llama-2-70b-hf)
model_name = os.environ.get("MODEL_NAME")

# Set OpenAI's API key to 'EMPTY' and API base URL to use vLLM's API server, or set them to OpenAI API key and base URL.
openai_api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
openai_api_base = os.environ.get("OPENAI_API_BASE") or "http://localhost:8000/v1"

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

print(f"Uploads={uploads}")
print(f"MODEL_DIR={model_dir}")
print(f"MODEL_NAME={model_name}")
print(f"USE_METAL={useMetal}")
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

# langchain.debug = True
langchain.verbose = True

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(vector_store.initRetrievers())
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
vector_store = VectorStorage(directory=uploads)


async def create_chain(
    messages: list[Message],
    model=model_dir,
    max_tokens_to_gen=2048,
    temperature=0.5,
    prompt: PromptTemplate = QA_PROMPT_TEMPLATE,
    file: RAGFile | None = None,
    key: str | None = None,
):
    print(
        f"Creating chain with key={key}, model={model}, max_tokens={max_tokens_to_gen}, temperature={temperature}, prompt={prompt}, file={file.title}"
    )

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

    retriever = await vector_store.getRetriever(os.path.join(file.username, file.filename))

    chat_memory = ChatMessageHistory()

    for message in messages:
        if message.role == Role.USER:
            chat_memory.add_user_message(message)
        elif message.role == Role.ASSISTANT:
            chat_memory.add_ai_message(message)
        elif message.role == Role.SYSTEM:
            chat_memory.add_message(message)

    memory = ConversationStringBufferMemory(
        llm=llm,
        chat_memory=chat_memory,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        prompt=SUMMARY_PROMPT_TEMPLATE,
        return_messages=False,
    )

    # memory = VectorStoreRetrieverMemory(
    #     input_key="question",
    #     output_key="answer",
    #     chat_memory=chat_memory,
    #     memory_key="chat_history",
    #     return_docs=False,  # Change this to False
    #     retriever=retriever,
    #     return_messages=True,
    #     prompt=SUMMARY_PROMPT_TEMPLATE
    # )

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
    chain: ConversationalRetrievalChain = await create_chain(
        file=request.file,
        messages=request.messages[:-1],
        model=request.model.id,
        max_tokens_to_gen=request.maxTokens,
        temperature=request.temperature,
        prompt=PromptTemplate.from_template(
            f"{B_INST}{B_SYS}{request.prompt.strip()}{E_SYS}{E_INST}"
        ),
    )

    # async for token in chain.astream(request.messages[-1].content):
    #     print(f"token={token}")

    json_string = json.dumps(
        {
            "question": request.messages[-1].content,
            # "chat_history": [message.content for message in request.messages[:-1]],
        }
    )
    return LangchainStreamingResponse(
        chain,
        config={
            "inputs": json_string,
            "callbacks": [
                StreamingStdOutCallbackHandler(),
                TokenStreamingCallbackHandler(output_key="answer"),
                SourceDocumentsStreamingCallbackHandler(),
            ],
        },
    )


app.include_router(router, tags=["chat"])


@app.get("/")
def root():
    return {"message": "Chatbot API"}


@app.get("/favicon.ico")
def favicon():
    file_name = "favicon.ico"
    file_path = os.path.join(app.root_path, "static", file_name)
    return FileResponse(
        path=file_path,
        headers={"Content-Disposition": "attachment; filename=" + file_name},
    )


@app.post("/log")
def log_message(log_message: LogMessage):
    try:
        with open("log.txt", "a") as log_file:
            log_file.write(log_message.message + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving log: {e}")
    return {"message": "Log saved successfully"}


@app.get("/models")
def get_models():
    # llama7B = AIModel(
    #     id="llama-2-7b-chat-ggml-q4_0",
    #     name="llama-2-7b-chat-ggml-q4_0",
    #     maxLength=2048,
    #     tokenLimit=2048,
    # )
    # llama13B = AIModel(
    #     id="llama-2-13b-chat-ggml-q4_0",
    #     name="llama-2-13b-chat-ggml-q4_0",
    #     maxLength=2048,
    #     tokenLimit=2048,
    # )
    llama70B = AIModel(
        id="llama-2-70b.Q5_K_M",
        name="llama-2-70b.Q5_K_M",
        maxLength=2048,
        tokenLimit=2048,
    )
    models = AIModels(models=[llama70B])
    return models


@app.get("/titles")
def getTitles():
    titles = RAGFiles(
        titles=[
            # RAGFile(
            #     versionId="d8ad3b1d-c33c-4524-9691-e93967d4d863",
            #     title="d8ad3b1d-c33c-4524-9691-e93967d4d863",
            #     state=State.Unavailable,
            # ),
            RAGFile(
                versionId=collection.name,
                title=collection.name,
                state=State.InProcess
                if collection.name in processing_books
                else State.Processed,
            )
            for collection in vector_store.list_collections()
            if collection.name != "langchain"
        ]
    )
    return titles


processing_books: list[str] = []
processing_books_lock = asyncio.Lock()

logging.basicConfig(level=logging.ERROR)


@app.post("/titleState")
async def getTitleState(request: GetRAGFileStateRequest):
    # FastAPI + Pydantic will throw a 422 Unprocessable Entity if the request isn't the right type.
    # try:
    logging.debug(f"Received getTitleState request: {request}")
    titleStateRequest: GetRAGFileStateRequest = request
    # except ValidationError as e:
    #     print(f"Error validating JSON: {e}")
    #     raise HTTPException(status_code=422, detail=str(e))
    # except json.JSONDecodeError as e:
    #     print(f"Error parsing JSON: {e}")
    #     raise HTTPException(status_code=422, detail="Invalid JSON format")
    # check to see if the book has already been processed.
    # return the proper State directly to response.
    matchingCollection = next(
        (
            x
            for x in vector_store.list_collections()
            if x.name == titleStateRequest.versionRef
        ),
        None,
    )
    print("Got a Title State request for version " + titleStateRequest.versionRef)
    if titleStateRequest.versionRef in processing_books:
        return {"message": State.InProcess}
    elif matchingCollection is not None:
        return {"message": State.Processed}
    else:
        return {"message": State.Unavailable}


@app.post("/processRAGFile")
async def processRAGFile(
    request: str = Form(...),
    files: List[UploadFile] = File(...),
):
    try:
        logging.debug(f"Received processBook request: {request}")
        # Parse the JSON string into a ProcessBookRequest object
        fileRAGRequest: ProcessRAGFileRequest = parse_obj_as(
            ProcessRAGFileRequest, json.loads(request)
        )
    except ValidationError as e:
        print(f"Error validating JSON: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        raise HTTPException(status_code=422, detail="Invalid JSON format")

    try:
        print(
            f"Processing file {fileRAGRequest.filename} for user {fileRAGRequest.username}."
        )
        # check to see if the file has already been processed.
        # write html to subfolder
        print(f"Writing file to path: {fileRAGRequest.username}/{fileRAGRequest.filename}...")

        for index, segment in enumerate(files):
            filename = segment.filename if segment.filename else str(index)
            subDir = f"{fileRAGRequest.username}"
            with open(os.path.join(subDir, filename), "wb") as htmlFile:
                htmlFile.write(await segment.read())

        # write metadata to subfolder
        print(f"Writing metadata to subfolder {fileRAGRequest.username}...")
        with open(os.path.join({fileRAGRequest.username}, "metadata.json"), "w") as metadataFile:
            metaData = {
                "filename": fileRAGRequest.filename,
                "username": fileRAGRequest.username,
                "processDate": datetime.now().isoformat(),
            }
            metadataFile.write(json.dumps(metaData))

        vector_store.retrievers[
            f"{fileRAGRequest.username}/{fileRAGRequest.filename}"
        ] = await vector_store.initRetriever(f"{fileRAGRequest.username}/{fileRAGRequest.filename}")

        return {
            "message": f"File {fileRAGRequest.filename} processed successfully."
        }
    except Exception as e:
        logging.error(f"Error processing book: {e}")
        return {"message": f"Error processing book: {e}"}


@app.exception_handler(HTTPException)
async def http_exception_handler(bookRequest: Request, exc: HTTPException):
    logging.error(f"HTTPException: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

