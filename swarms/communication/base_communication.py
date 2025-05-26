from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
from enum import Enum
from dataclasses import dataclass
from pathlib import Path


class MessageType(Enum):
    """Enum for different types of messages in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """Data class representing a message in the conversation."""

    role: str
    content: Union[str, dict, list]
    timestamp: Optional[str] = None
    message_type: Optional[MessageType] = None
    metadata: Optional[Dict] = None
    token_count: Optional[int] = None


class BaseCommunication(ABC):
    """
    Abstract base class defining the interface for conversation implementations.
    This class provides the contract that all conversation implementations must follow.

    Attributes:
        system_prompt (Optional[str]): The system prompt for the conversation.
        time_enabled (bool): Flag to enable time tracking for messages.
        autosave (bool): Flag to enable automatic saving of conversation history.
        save_filepath (str): File path for saving the conversation history.
        tokenizer (Any): Tokenizer for counting tokens in messages.
        context_length (int): Maximum number of tokens allowed in the conversation history.
        rules (str): Rules for the conversation.
        custom_rules_prompt (str): Custom prompt for rules.
        user (str): The user identifier for messages.
        auto_save (bool): Flag to enable auto-saving of conversation history.
        save_as_yaml (bool): Flag to save conversation history as YAML.
        save_as_json_bool (bool): Flag to save conversation history as JSON.
        token_count (bool): Flag to enable token counting for messages.
        cache_enabled (bool): Flag to enable prompt caching.
    """

    @staticmethod
    def get_default_db_path(db_name: str) -> Path:
        """Calculate the default database path in user's home directory.

        Args:
            db_name (str): Name of the database file (e.g. 'conversations.db')

        Returns:
            Path: Path object pointing to the database location
        """
        # Get user's home directory
        home = Path.home()

        # Create .swarms directory if it doesn't exist
        swarms_dir = home / ".swarms" / "db"
        swarms_dir.mkdir(parents=True, exist_ok=True)

        return swarms_dir / db_name

    @abstractmethod
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        time_enabled: bool = False,
        autosave: bool = False,
        save_filepath: str = None,
        tokenizer: Any = None,
        context_length: int = 8192,
        rules: str = None,
        custom_rules_prompt: str = None,
        user: str = "User:",
        auto_save: bool = True,
        save_as_yaml: bool = True,
        save_as_json_bool: bool = False,
        token_count: bool = True,
        cache_enabled: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the communication interface."""
        pass

    @abstractmethod
    def add(
        self,
        role: str,
        content: Union[str, dict, list],
        message_type: Optional[MessageType] = None,
        metadata: Optional[Dict] = None,
        token_count: Optional[int] = None,
    ) -> int:
        """Add a message to the conversation history."""
        pass

    @abstractmethod
    def batch_add(self, messages: List[Message]) -> List[int]:
        """Add multiple messages to the conversation history."""
        pass

    @abstractmethod
    def delete(self, index: str):
        """Delete a message from the conversation history."""
        pass

    @abstractmethod
    def update(
        self, index: str, role: str, content: Union[str, dict]
    ):
        """Update a message in the conversation history."""
        pass

    @abstractmethod
    def query(self, index: str) -> Dict:
        """Query a message in the conversation history."""
        pass

    @abstractmethod
    def search(self, keyword: str) -> List[Dict]:
        """Search for messages containing a keyword."""
        pass

    @abstractmethod
    def get_str(self) -> str:
        """Get the conversation history as a string."""
        pass

    @abstractmethod
    def display_conversation(self, detailed: bool = False):
        """Display the conversation history."""
        pass

    @abstractmethod
    def export_conversation(self, filename: str):
        """Export the conversation history to a file."""
        pass

    @abstractmethod
    def import_conversation(self, filename: str):
        """Import a conversation history from a file."""
        pass

    @abstractmethod
    def count_messages_by_role(self) -> Dict[str, int]:
        """Count messages by role."""
        pass

    @abstractmethod
    def return_history_as_string(self) -> str:
        """Return the conversation history as a string."""
        pass

    @abstractmethod
    def get_messages(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict]:
        """Get messages with optional pagination."""
        pass

    @abstractmethod
    def clear(self):
        """Clear the conversation history."""
        pass

    @abstractmethod
    def to_dict(self) -> List[Dict]:
        """Convert the conversation history to a dictionary."""
        pass

    @abstractmethod
    def to_json(self) -> str:
        """Convert the conversation history to a JSON string."""
        pass

    @abstractmethod
    def to_yaml(self) -> str:
        """Convert the conversation history to a YAML string."""
        pass

    @abstractmethod
    def save_as_json(self, filename: str):
        """Save the conversation history as a JSON file."""
        pass

    @abstractmethod
    def load_from_json(self, filename: str):
        """Load the conversation history from a JSON file."""
        pass

    @abstractmethod
    def save_as_yaml(self, filename: str):
        """Save the conversation history as a YAML file."""
        pass

    @abstractmethod
    def load_from_yaml(self, filename: str):
        """Load the conversation history from a YAML file."""
        pass

    @abstractmethod
    def get_last_message(self) -> Optional[Dict]:
        """Get the last message from the conversation history."""
        pass

    @abstractmethod
    def get_last_message_as_string(self) -> str:
        """Get the last message as a formatted string."""
        pass

    @abstractmethod
    def get_messages_by_role(self, role: str) -> List[Dict]:
        """Get all messages from a specific role."""
        pass

    @abstractmethod
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict:
        """Get statistics about the conversation."""
        pass

    @abstractmethod
    def get_conversation_id(self) -> str:
        """Get the current conversation ID."""
        pass

    @abstractmethod
    def start_new_conversation(self) -> str:
        """Start a new conversation and return its ID."""
        pass

    @abstractmethod
    def delete_current_conversation(self) -> bool:
        """Delete the current conversation."""
        pass

    @abstractmethod
    def search_messages(self, query: str) -> List[Dict]:
        """Search for messages containing specific text."""
        pass

    @abstractmethod
    def update_message(
        self,
        message_id: int,
        content: Union[str, dict, list],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Update an existing message."""
        pass

    @abstractmethod
    def get_conversation_metadata_dict(self) -> Dict:
        """Get detailed metadata about the conversation."""
        pass

    @abstractmethod
    def get_conversation_timeline_dict(self) -> Dict[str, List[Dict]]:
        """Get the conversation organized by timestamps."""
        pass

    @abstractmethod
    def get_conversation_by_role_dict(self) -> Dict[str, List[Dict]]:
        """Get the conversation organized by roles."""
        pass

    @abstractmethod
    def get_conversation_as_dict(self) -> Dict:
        """Get the entire conversation as a dictionary with messages and metadata."""
        pass

    @abstractmethod
    def truncate_memory_with_tokenizer(self):
        """Truncate the conversation history based on token count."""
        pass
