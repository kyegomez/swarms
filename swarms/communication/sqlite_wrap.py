import sqlite3
import json
import datetime
from typing import List, Optional, Union, Dict
from pathlib import Path
import threading
from contextlib import contextmanager
import logging
from dataclasses import dataclass
from enum import Enum
import uuid
import yaml

try:
    from loguru import logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False


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

    class Config:
        arbitrary_types_allowed = True


class SQLiteConversation:
    """
    A production-grade SQLite wrapper class for managing conversation history.
    This class provides persistent storage for conversations with various features
    like message tracking, timestamps, and metadata support.

    Attributes:
        db_path (str): Path to the SQLite database file
        table_name (str): Name of the table to store conversations
        enable_timestamps (bool): Whether to track message timestamps
        enable_logging (bool): Whether to enable logging
        use_loguru (bool): Whether to use loguru for logging
        max_retries (int): Maximum number of retries for database operations
        connection_timeout (float): Timeout for database connections
        current_conversation_id (str): Current active conversation ID
    """

    def __init__(
        self,
        db_path: str = "conversations.db",
        table_name: str = "conversations",
        enable_timestamps: bool = True,
        enable_logging: bool = True,
        use_loguru: bool = True,
        max_retries: int = 3,
        connection_timeout: float = 5.0,
        **kwargs,
    ):
        """
        Initialize the SQLite conversation manager.

        Args:
            db_path (str): Path to the SQLite database file
            table_name (str): Name of the table to store conversations
            enable_timestamps (bool): Whether to track message timestamps
            enable_logging (bool): Whether to enable logging
            use_loguru (bool): Whether to use loguru for logging
            max_retries (int): Maximum number of retries for database operations
            connection_timeout (float): Timeout for database connections
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.enable_timestamps = enable_timestamps
        self.enable_logging = enable_logging
        self.use_loguru = use_loguru and LOGURU_AVAILABLE
        self.max_retries = max_retries
        self.connection_timeout = connection_timeout
        self._lock = threading.Lock()
        self.current_conversation_id = (
            self._generate_conversation_id()
        )

        # Setup logging
        if self.enable_logging:
            if self.use_loguru:
                self.logger = logger
            else:
                self.logger = logging.getLogger(__name__)
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        # Initialize database
        self._init_db()

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID using UUID and timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"conv_{timestamp}_{unique_id}"

    def start_new_conversation(self) -> str:
        """
        Start a new conversation and return its ID.

        Returns:
            str: The new conversation ID
        """
        self.current_conversation_id = (
            self._generate_conversation_id()
        )
        return self.current_conversation_id

    def _init_db(self):
        """Initialize the database and create necessary tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT,
                    message_type TEXT,
                    metadata TEXT,
                    token_count INTEGER,
                    conversation_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with retry logic."""
        conn = None
        for attempt in range(self.max_retries):
            try:
                conn = sqlite3.connect(
                    str(self.db_path), timeout=self.connection_timeout
                )
                conn.row_factory = sqlite3.Row
                yield conn
                break
            except sqlite3.Error as e:
                if attempt == self.max_retries - 1:
                    raise
                if self.enable_logging:
                    self.logger.warning(
                        f"Database connection attempt {attempt + 1} failed: {e}"
                    )
            finally:
                if conn:
                    conn.close()

    def add(
        self,
        role: str,
        content: Union[str, dict, list],
        message_type: Optional[MessageType] = None,
        metadata: Optional[Dict] = None,
        token_count: Optional[int] = None,
    ) -> int:
        """
        Add a message to the current conversation.

        Args:
            role (str): The role of the speaker
            content (Union[str, dict, list]): The content of the message
            message_type (Optional[MessageType]): Type of the message
            metadata (Optional[Dict]): Additional metadata for the message
            token_count (Optional[int]): Number of tokens in the message

        Returns:
            int: The ID of the inserted message
        """
        timestamp = (
            datetime.datetime.now().isoformat()
            if self.enable_timestamps
            else None
        )

        if isinstance(content, (dict, list)):
            content = json.dumps(content)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {self.table_name} 
                (role, content, timestamp, message_type, metadata, token_count, conversation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    role,
                    content,
                    timestamp,
                    message_type.value if message_type else None,
                    json.dumps(metadata) if metadata else None,
                    token_count,
                    self.current_conversation_id,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def batch_add(self, messages: List[Message]) -> List[int]:
        """
        Add multiple messages to the current conversation.

        Args:
            messages (List[Message]): List of messages to add

        Returns:
            List[int]: List of inserted message IDs
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            message_ids = []

            for message in messages:
                content = message.content
                if isinstance(content, (dict, list)):
                    content = json.dumps(content)

                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name} 
                    (role, content, timestamp, message_type, metadata, token_count, conversation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        message.role,
                        content,
                        (
                            message.timestamp.isoformat()
                            if message.timestamp
                            else None
                        ),
                        (
                            message.message_type.value
                            if message.message_type
                            else None
                        ),
                        (
                            json.dumps(message.metadata)
                            if message.metadata
                            else None
                        ),
                        message.token_count,
                        self.current_conversation_id,
                    ),
                )
                message_ids.append(cursor.lastrowid)

            conn.commit()
            return message_ids

    def get_str(self) -> str:
        """
        Get the current conversation history as a formatted string.

        Returns:
            str: Formatted conversation history
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
            """,
                (self.current_conversation_id,),
            )

            messages = []
            for row in cursor.fetchall():
                content = row["content"]
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                timestamp = (
                    f"[{row['timestamp']}] "
                    if row["timestamp"]
                    else ""
                )
                messages.append(
                    f"{timestamp}{row['role']}: {content}"
                )

            return "\n".join(messages)

    def get_messages(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get messages from the current conversation with optional pagination.

        Args:
            limit (Optional[int]): Maximum number of messages to return
            offset (Optional[int]): Number of messages to skip

        Returns:
            List[Dict]: List of message dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
            """
            params = [self.current_conversation_id]

            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            if offset is not None:
                query += " OFFSET ?"
                params.append(offset)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def delete_current_conversation(self) -> bool:
        """
        Delete the current conversation.

        Returns:
            bool: True if deletion was successful
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE conversation_id = ?",
                (self.current_conversation_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_message(
        self,
        message_id: int,
        content: Union[str, dict, list],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Update an existing message in the current conversation.

        Args:
            message_id (int): ID of the message to update
            content (Union[str, dict, list]): New content for the message
            metadata (Optional[Dict]): New metadata for the message

        Returns:
            bool: True if update was successful
        """
        if isinstance(content, (dict, list)):
            content = json.dumps(content)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                UPDATE {self.table_name}
                SET content = ?, metadata = ?
                WHERE id = ? AND conversation_id = ?
            """,
                (
                    content,
                    json.dumps(metadata) if metadata else None,
                    message_id,
                    self.current_conversation_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def search_messages(self, query: str) -> List[Dict]:
        """
        Search for messages containing specific text in the current conversation.

        Args:
            query (str): Text to search for

        Returns:
            List[Dict]: List of matching messages
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE conversation_id = ? AND content LIKE ?
            """,
                (self.current_conversation_id, f"%{query}%"),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict:
        """
        Get statistics about the current conversation.

        Returns:
            Dict: Statistics about the conversation
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(DISTINCT role) as unique_roles,
                    SUM(token_count) as total_tokens,
                    MIN(timestamp) as first_message,
                    MAX(timestamp) as last_message
                FROM {self.table_name}
                WHERE conversation_id = ?
            """,
                (self.current_conversation_id,),
            )
            return dict(cursor.fetchone())

    def clear_all(self) -> bool:
        """
        Clear all messages from the database.

        Returns:
            bool: True if clearing was successful
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.table_name}")
            conn.commit()
            return True

    def get_conversation_id(self) -> str:
        """
        Get the current conversation ID.

        Returns:
            str: The current conversation ID
        """
        return self.current_conversation_id

    def to_dict(self) -> List[Dict]:
        """
        Convert the current conversation to a list of dictionaries.

        Returns:
            List[Dict]: List of message dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT role, content, timestamp, message_type, metadata, token_count
                FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
            """,
                (self.current_conversation_id,),
            )

            messages = []
            for row in cursor.fetchall():
                content = row["content"]
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                message = {"role": row["role"], "content": content}

                if row["timestamp"]:
                    message["timestamp"] = row["timestamp"]
                if row["message_type"]:
                    message["message_type"] = row["message_type"]
                if row["metadata"]:
                    message["metadata"] = json.loads(row["metadata"])
                if row["token_count"]:
                    message["token_count"] = row["token_count"]

                messages.append(message)

            return messages

    def to_json(self) -> str:
        """
        Convert the current conversation to a JSON string.

        Returns:
            str: JSON string representation of the conversation
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_yaml(self) -> str:
        """
        Convert the current conversation to a YAML string.

        Returns:
            str: YAML string representation of the conversation
        """
        return yaml.dump(self.to_dict())

    def save_as_json(self, filename: str) -> bool:
        """
        Save the current conversation to a JSON file.

        Args:
            filename (str): Path to save the JSON file

        Returns:
            bool: True if save was successful
        """
        try:
            with open(filename, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Failed to save conversation to JSON: {e}"
                )
            return False

    def save_as_yaml(self, filename: str) -> bool:
        """
        Save the current conversation to a YAML file.

        Args:
            filename (str): Path to save the YAML file

        Returns:
            bool: True if save was successful
        """
        try:
            with open(filename, "w") as f:
                yaml.dump(self.to_dict(), f)
            return True
        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Failed to save conversation to YAML: {e}"
                )
            return False

    def load_from_json(self, filename: str) -> bool:
        """
        Load a conversation from a JSON file.

        Args:
            filename (str): Path to the JSON file

        Returns:
            bool: True if load was successful
        """
        try:
            with open(filename, "r") as f:
                messages = json.load(f)

            # Start a new conversation
            self.start_new_conversation()

            # Add all messages
            for message in messages:
                self.add(
                    role=message["role"],
                    content=message["content"],
                    message_type=(
                        MessageType(message["message_type"])
                        if "message_type" in message
                        else None
                    ),
                    metadata=message.get("metadata"),
                    token_count=message.get("token_count"),
                )
            return True
        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Failed to load conversation from JSON: {e}"
                )
            return False

    def load_from_yaml(self, filename: str) -> bool:
        """
        Load a conversation from a YAML file.

        Args:
            filename (str): Path to the YAML file

        Returns:
            bool: True if load was successful
        """
        try:
            with open(filename, "r") as f:
                messages = yaml.safe_load(f)

            # Start a new conversation
            self.start_new_conversation()

            # Add all messages
            for message in messages:
                self.add(
                    role=message["role"],
                    content=message["content"],
                    message_type=(
                        MessageType(message["message_type"])
                        if "message_type" in message
                        else None
                    ),
                    metadata=message.get("metadata"),
                    token_count=message.get("token_count"),
                )
            return True
        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Failed to load conversation from YAML: {e}"
                )
            return False

    def get_last_message(self) -> Optional[Dict]:
        """
        Get the last message from the current conversation.

        Returns:
            Optional[Dict]: The last message or None if conversation is empty
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT role, content, timestamp, message_type, metadata, token_count
                FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT 1
            """,
                (self.current_conversation_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            content = row["content"]
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass

            message = {"role": row["role"], "content": content}

            if row["timestamp"]:
                message["timestamp"] = row["timestamp"]
            if row["message_type"]:
                message["message_type"] = row["message_type"]
            if row["metadata"]:
                message["metadata"] = json.loads(row["metadata"])
            if row["token_count"]:
                message["token_count"] = row["token_count"]

            return message

    def get_last_message_as_string(self) -> str:
        """
        Get the last message as a formatted string.

        Returns:
            str: Formatted string of the last message
        """
        last_message = self.get_last_message()
        if not last_message:
            return ""

        timestamp = (
            f"[{last_message['timestamp']}] "
            if "timestamp" in last_message
            else ""
        )
        return f"{timestamp}{last_message['role']}: {last_message['content']}"

    def count_messages_by_role(self) -> Dict[str, int]:
        """
        Count messages by role in the current conversation.

        Returns:
            Dict[str, int]: Dictionary with role counts
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT role, COUNT(*) as count
                FROM {self.table_name}
                WHERE conversation_id = ?
                GROUP BY role
            """,
                (self.current_conversation_id,),
            )

            return {
                row["role"]: row["count"] for row in cursor.fetchall()
            }

    def get_messages_by_role(self, role: str) -> List[Dict]:
        """
        Get all messages from a specific role in the current conversation.

        Args:
            role (str): Role to filter messages by

        Returns:
            List[Dict]: List of messages from the specified role
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT role, content, timestamp, message_type, metadata, token_count
                FROM {self.table_name}
                WHERE conversation_id = ? AND role = ?
                ORDER BY id ASC
            """,
                (self.current_conversation_id, role),
            )

            messages = []
            for row in cursor.fetchall():
                content = row["content"]
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                message = {"role": row["role"], "content": content}

                if row["timestamp"]:
                    message["timestamp"] = row["timestamp"]
                if row["message_type"]:
                    message["message_type"] = row["message_type"]
                if row["metadata"]:
                    message["metadata"] = json.loads(row["metadata"])
                if row["token_count"]:
                    message["token_count"] = row["token_count"]

                messages.append(message)

            return messages

    def get_conversation_summary(self) -> Dict:
        """
        Get a summary of the current conversation.

        Returns:
            Dict: Summary of the conversation including message counts, roles, and time range
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(DISTINCT role) as unique_roles,
                    MIN(timestamp) as first_message_time,
                    MAX(timestamp) as last_message_time,
                    SUM(token_count) as total_tokens
                FROM {self.table_name}
                WHERE conversation_id = ?
            """,
                (self.current_conversation_id,),
            )

            row = cursor.fetchone()
            return {
                "conversation_id": self.current_conversation_id,
                "total_messages": row["total_messages"],
                "unique_roles": row["unique_roles"],
                "first_message_time": row["first_message_time"],
                "last_message_time": row["last_message_time"],
                "total_tokens": row["total_tokens"],
                "roles": self.count_messages_by_role(),
            }
