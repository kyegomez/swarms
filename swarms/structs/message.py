import datetime
from typing import Dict, Optional


class Message:
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

    def __init__(
        self,
        sender: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
    ):
        self.timestamp: datetime.datetime = datetime.datetime.now()
        self.sender: str = sender
        self.content: str = content
        self.metadata: Dict[str, str] = metadata or {}

    def __repr__(self) -> str:
        """
        __repr__ means...
        """
        return f"{self.timestamp} - {self.sender}: {self.content}"
