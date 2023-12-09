import os
import re
from multiprocessing import Process
from tempfile import NamedTemporaryFile

from typing import List, TypedDict
import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse

from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.container import agent_manager, file_handler, reload_dirs, templates, uploader
from api.worker import get_task_result, start_worker, task_execute


app = FastAPI()

app.mount("/static", StaticFiles(directory=uploader.path), name="static")


class ExecuteRequest(BaseModel):
    session: str
    prompt: str
    files: List[str]


class ExecuteResponse(TypedDict):
    answer: str
    files: List[str]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post("/upload")
async def create_upload_file(files: List[UploadFile]):
    urls = []
    for file in files:
        extension = "." + file.filename.split(".")[-1]
        with NamedTemporaryFile(suffix=extension) as tmp_file:
            tmp_file.write(file.file.read())
            tmp_file.flush()
            urls.append(uploader.upload(tmp_file.name))
    return {"urls": urls}


@app.post("/api/execute")
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    query = request.prompt
    files = request.files
    session = request.session

    executor = agent_manager.create_executor(session)

    promptedQuery = "\n".join([file_handler.handle(file) for file in files])
    promptedQuery += query

    try:
        res = executor({"input": promptedQuery})
    except Exception as e:
        return {"answer": str(e), "files": []}

    files = re.findall(r"\[file://\S*\]", res["output"])
    files = [file[1:-1].split("file://")[1] for file in files]

    return {
        "answer": res["output"],
        "files": [uploader.upload(file) for file in files],
    }


@app.post("/api/execute/async")
async def execute_async(request: ExecuteRequest):
    query = request.prompt
    files = request.files
    session = request.session

    promptedQuery = "\n".join([file_handler.handle(file) for file in files])
    promptedQuery += query

    execution = task_execute.delay(session, promptedQuery)
    return {"id": execution.id}


@app.get("/api/execute/async/{execution_id}")
async def execute_async(execution_id: str):
    execution = get_task_result(execution_id)

    result = {}
    if execution.status == "SUCCESS" and execution.result:
        output = execution.result.get("output", "")
        files = re.findall(r"\[file://\S*\]", output)
        files = [file[1:-1].split("file://")[1] for file in files]
        result = {
            "answer": output,
            "files": [uploader.upload(file) for file in files],
        }

    return {
        "status": execution.status,
        "info": execution.info,
        "result": result,
    }


def serve():
    p = Process(target=start_worker, args=[])
    p.start()
    uvicorn.run("api.main:app", host="0.0.0.0", port=os.getenv["EVAL_PORT"])


def dev():
    p = Process(target=start_worker, args=[])
    p.start()
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=os.getenv["EVAL_PORT"],
        reload=True,
        reload_dirs=reload_dirs,
    )
