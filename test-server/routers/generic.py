from fastapi import APIRouter
from fastapi import Request
from fastapi import FastAPI, Request
from routers import openai
import logging

logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{rest_of_path:path}", status_code=500)
@router.post("/{rest_of_path:path}", status_code=500)
@router.put("/{rest_of_path:path}", status_code=500)
@router.patch("/{rest_of_path:path}", status_code=500)
@router.delete("/{rest_of_path:path}", status_code=500)
def catch_all(request: Request):

    path = request.path_params["rest_of_path"]

    logger.info("Method not mocked")
    logger.info(f"Path: {path}")
    logger.info(f"Method: {request.method}")

    return {
        "message": "This path is not mocked",
        "path": path,
        "method": request.method,
    }
