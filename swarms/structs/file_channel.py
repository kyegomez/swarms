import re
from pathlib import Path
from typing import Any, Dict, List

from swarms.structs.comms_channel import CommsChannel


class FileChannel(CommsChannel):
    """A markdown or text file where each message starts with a ## @agent_name line.

    The agent_name is everything after the @ on that line, so names with spaces
    work. The body is the text up to the next such line, kept as written,
    so markdown headings inside a message survive.

    Args:
        path: A .md or .txt file, created on the first post.

    Raises:
        ValueError: If the path has another extension.

    Example:
        ## @user

        Write a market report.

        ## @Researcher

        The market grew 31% last year ...
    """

    HEADER = re.compile(r"^## @(.+?)[ \t]*$", re.MULTILINE)

    EXTENSIONS = (".md", ".txt")

    def __init__(self, path: str):
        """
        Initialize a FileChannel instance.

        Args:
            path (str): Path to the .md or .txt file. The file will be created on the first `post` if it doesn't exist.

        Raises:
            ValueError: If the file does not have a .md or .txt extension.
        """
        self.path = Path(path)
        if self.path.suffix.lower() not in self.EXTENSIONS:
            raise ValueError(
                f"FileChannel takes a .md or .txt file, got '{self.path.name}'"
            )

    def post(self, agent_name: str, content: str) -> None:
        """
        Append a message to the file with the given agent_name and content.

        Each message is prepended with "## @agent_name" to denote the sender's agent_name.

        Args:
            agent_name (str): The agent_name/name of the sender, e.g. "user" or "assistant".
            content (str): The content/message to be added.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"\n## @{agent_name}\n\n{str(content).strip()}\n")

    def messages(self) -> List[Dict[str, Any]]:
        """
        Retrieve a list of all messages in the file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries in the form {'agent_name': ..., 'content': ...},
                                  with each message as parsed from the markdown or text file.
        """
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        parts = self.HEADER.split(text)[1:]
        return [
            {"agent_name": agent_name, "content": body.strip()}
            for agent_name, body in zip(parts[::2], parts[1::2])
        ]

    def clear(self) -> None:
        """
        Delete the current file, clearing all messages.

        If the file does not exist, nothing happens.
        """
        self.path.unlink(missing_ok=True)
