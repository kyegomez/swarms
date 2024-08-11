from fastapi import APIRouter
from .mock.openai_mock import success_chat_completion

router = APIRouter()


@router.post("/success/chat/completions")
def success_chat_completions():
    return success_chat_completion
