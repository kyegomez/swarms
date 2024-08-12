from typing import Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    Represents a message with timestamp and optional metadata.

    Usage
    --------------
    mes = Message(
        sender = "Kye",
        content = "message"
    )

    print(mes)
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    sender: str
    content: str
    metadata: Optional[Dict[str, str]] = {}

    def __repr__(self) -> str:
        """
        __repr__ means...
        """
        return f"{self.timestamp} - {self.sender}: {self.content}"
