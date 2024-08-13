try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

from pydantic import BaseModel
from swarms.prompts import QA_PROMPT_TEMPLATE_STR as DefaultSystemPrompt

class AIModel(BaseModel):
    id: str
    name: str
    maxLength: int
    tokenLimit: int


class AIModels(BaseModel):
    models: list[AIModel]


class State(StrEnum):
    Unavailable = "Unavailable"
    InProcess = "InProcess"
    Processed = "Processed"


class RAGFile(BaseModel):
    filename: str
    title: str
    username: str
    state: State = State.Unavailable


class RAGFiles(BaseModel):
    files: list[RAGFile]


class Role(StrEnum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    id: str
    model: AIModel = AIModel(
        id="llama-2-70b.Q5_K_M",
        name="llama-2-70b.Q5_K_M",
        maxLength=2048,
        tokenLimit=2048,
    )
    messages: list[Message] = [
        Message(role=Role.SYSTEM, content="Hello, how may I help you?"),
        Message(role=Role.USER, content=""),
    ]
    maxTokens: int = 2048
    temperature: float = 0
    prompt: str = DefaultSystemPrompt
    file: RAGFile = RAGFile(filename="None", title="None", username="None")


class LogMessage(BaseModel):
    message: str


class ConversationRequest(BaseModel):
    id: str
    name: str
    title: RAGFile
    messages: list[Message]
    model: AIModel
    prompt: str
    temperature: float
    folderId: str | None = None


class ProcessRAGFileRequest(BaseModel):
    filename: str
    username: str


class GetRAGFileStateRequest(BaseModel):
    filename: str
    username: str