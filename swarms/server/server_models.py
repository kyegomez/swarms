""" Chatbot Server API Models """
try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum

from pydantic import BaseModel
from swarms.prompts import QA_PROMPT_TEMPLATE_STR as DefaultSystemPrompt

class AIModel(BaseModel):
    """ Defines the model a user selected. """
    id: str
    name: str
    maxLength: int
    tokenLimit: int


class State(StrEnum):
    """ State of RAGFile that's been uploaded. """
    UNAVAILABLE = "UNAVAILABLE"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"


class RAGFile(BaseModel):
    """ Defines a file uploaded by the users for RAG processing. """
    filename: str
    title: str
    username: str
    state: State = State.UNAVAILABLE


class RAGFiles(BaseModel):
    """ Defines a list of RAGFile objects. """
    files: list[RAGFile]


class Role(StrEnum):
    """ The role of a message in a conversation. """
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class Message(BaseModel):
    """ Defines the type of a Message with a role and content. """
    role: Role
    content: str


class ChatRequest(BaseModel):
    """ The model for a ChatRequest expected by the Chatbot Chat POST endpoint. """
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
