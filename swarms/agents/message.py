import datetime


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

    def __init__(self, sender, content, metadata=None):
        self.timestamp = datetime.datetime.now()
        self.sender = sender
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        """
        __repr__ means...
        """
        return f"{self.timestamp} - {self.sender}: {self.content}"
