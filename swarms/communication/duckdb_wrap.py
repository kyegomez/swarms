import datetime
import json
import logging
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from swarms.communication.base_communication import (
    BaseCommunication,
    Message,
    MessageType,
)

try:
    from loguru import logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


class DuckDBConversation(BaseCommunication):
    """
    A production-grade DuckDB wrapper class for managing conversation history.
    This class provides persistent storage for conversations with various features
    like message tracking, timestamps, and metadata support.

    Attributes:
        db_path (str): Path to the DuckDB database file
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
        db_path: Union[str, Path] = None,
        table_name: str = "conversations",
        enable_timestamps: bool = True,
        enable_logging: bool = True,
        use_loguru: bool = True,
        max_retries: int = 3,
        connection_timeout: float = 5.0,
        *args,
        **kwargs,
    ):
        # Lazy load duckdb with auto-installation
        try:
            import duckdb

            self.duckdb = duckdb
            self.duckdb_available = True
        except ImportError:
            # Auto-install duckdb if not available
            print("📦 DuckDB not found. Installing automatically...")
            try:
                import subprocess
                import sys

                # Install duckdb
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "duckdb"]
                )
                print("✅ DuckDB installed successfully!")

                # Try importing again
                import duckdb

                self.duckdb = duckdb
                self.duckdb_available = True
                print("✅ DuckDB loaded successfully!")

            except Exception as e:
                raise ImportError(
                    f"Failed to auto-install DuckDB. Please install manually with 'pip install duckdb': {e}"
                )

        super().__init__(
            system_prompt=system_prompt,
            time_enabled=time_enabled,
            autosave=autosave,
            save_filepath=save_filepath,
            tokenizer=tokenizer,
            context_length=context_length,
            rules=rules,
            custom_rules_prompt=custom_rules_prompt,
            user=user,
            auto_save=auto_save,
            save_as_yaml=save_as_yaml,
            save_as_json_bool=save_as_json_bool,
            token_count=token_count,
            cache_enabled=cache_enabled,
        )

        # Calculate default db_path if not provided
        if db_path is None:
            db_path = self.get_default_db_path("conversations.duckdb")
        self.db_path = Path(db_path)

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.table_name = table_name
        self.enable_timestamps = enable_timestamps
        self.enable_logging = enable_logging
        self.use_loguru = use_loguru and LOGURU_AVAILABLE
        self.max_retries = max_retries
        self.connection_timeout = connection_timeout
        self.current_conversation_id = None
        self._lock = threading.Lock()
        self.tokenizer = tokenizer

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

        self._init_db()
        self.start_new_conversation()

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
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id BIGINT PRIMARY KEY,
                    role VARCHAR NOT NULL,
                    content VARCHAR NOT NULL,
                    timestamp TIMESTAMP,
                    message_type VARCHAR,
                    metadata VARCHAR,
                    token_count INTEGER,
                    conversation_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with retry logic."""
        conn = None
        for attempt in range(self.max_retries):
            try:
                conn = self.duckdb.connect(str(self.db_path))
                yield conn
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                if self.enable_logging:
                    self.logger.warning(
                        f"Database connection attempt {attempt + 1} failed: {e}"
                    )
                if conn:
                    conn.close()
                    conn = None

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
            # Get the next ID
            result = conn.execute(
                f"SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM {self.table_name}"
            ).fetchone()
            next_id = result[0]

            # Insert the message
            conn.execute(
                f"""
                INSERT INTO {self.table_name} 
                (id, role, content, timestamp, message_type, metadata, token_count, conversation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    next_id,
                    role,
                    content,
                    timestamp,
                    message_type.value if message_type else None,
                    json.dumps(metadata) if metadata else None,
                    token_count,
                    self.current_conversation_id,
                ),
            )
            return next_id

    def batch_add(self, messages: List[Message]) -> List[int]:
        """
        Add multiple messages to the current conversation.

        Args:
            messages (List[Message]): List of messages to add

        Returns:
            List[int]: List of inserted message IDs
        """
        with self._get_connection() as conn:
            message_ids = []

            # Get the starting ID
            result = conn.execute(
                f"SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM {self.table_name}"
            ).fetchone()
            next_id = result[0]

            for i, message in enumerate(messages):
                content = message.content
                if isinstance(content, (dict, list)):
                    content = json.dumps(content)

                conn.execute(
                    f"""
                    INSERT INTO {self.table_name} 
                    (id, role, content, timestamp, message_type, metadata, token_count, conversation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        next_id + i,
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
                message_ids.append(next_id + i)

            return message_ids

    def get_str(self) -> str:
        """
        Get the current conversation history as a formatted string.

        Returns:
            str: Formatted conversation history
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
            """,
                (self.current_conversation_id,),
            ).fetchall()

            messages = []
            for row in result:
                content = row[2]  # content column
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                timestamp = (
                    f"[{row[3]}] " if row[3] else ""
                )  # timestamp column
                messages.append(
                    f"{timestamp}{row[1]}: {content}"
                )  # role column

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

            result = conn.execute(query, params).fetchall()
            messages = []
            for row in result:
                content = row[2]  # content column
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                message = {
                    "role": row[1],  # role column
                    "content": content,
                }

                if row[3]:  # timestamp column
                    message["timestamp"] = row[3]
                if row[4]:  # message_type column
                    message["message_type"] = row[4]
                if row[5]:  # metadata column
                    message["metadata"] = json.loads(row[5])
                if row[6]:  # token_count column
                    message["token_count"] = row[6]

                messages.append(message)

            return messages

    def delete_current_conversation(self) -> bool:
        """
        Delete the current conversation.

        Returns:
            bool: True if deletion was successful
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"DELETE FROM {self.table_name} WHERE conversation_id = ?",
                (self.current_conversation_id,),
            )
            return result.rowcount > 0

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
            result = conn.execute(
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
            return result.rowcount > 0

    def search_messages(self, query: str) -> List[Dict]:
        """
        Search for messages containing specific text in the current conversation.

        Args:
            query (str): Text to search for

        Returns:
            List[Dict]: List of matching messages
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE conversation_id = ? AND content LIKE ?
            """,
                (self.current_conversation_id, f"%{query}%"),
            ).fetchall()

            messages = []
            for row in result:
                content = row[2]  # content column
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                message = {
                    "role": row[1],  # role column
                    "content": content,
                }

                if row[3]:  # timestamp column
                    message["timestamp"] = row[3]
                if row[4]:  # message_type column
                    message["message_type"] = row[4]
                if row[5]:  # metadata column
                    message["metadata"] = json.loads(row[5])
                if row[6]:  # token_count column
                    message["token_count"] = row[6]

                messages.append(message)

            return messages

    def get_statistics(self) -> Dict:
        """
        Get statistics about the current conversation.

        Returns:
            Dict: Statistics about the conversation
        """
        with self._get_connection() as conn:
            result = conn.execute(
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
            ).fetchone()

            return {
                "total_messages": result[0],
                "unique_roles": result[1],
                "total_tokens": result[2],
                "first_message": result[3],
                "last_message": result[4],
            }

    def clear_all(self) -> bool:
        """
        Clear all messages from the database.

        Returns:
            bool: True if clearing was successful
        """
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
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
            result = conn.execute(
                f"""
                SELECT role, content, timestamp, message_type, metadata, token_count
                FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
            """,
                (self.current_conversation_id,),
            ).fetchall()

            messages = []
            for row in result:
                content = row[1]  # content column
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                message = {
                    "role": row[0],
                    "content": content,
                }  # role column

                if row[2]:  # timestamp column
                    message["timestamp"] = row[2]
                if row[3]:  # message_type column
                    message["message_type"] = row[3]
                if row[4]:  # metadata column
                    message["metadata"] = json.loads(row[4])
                if row[5]:  # token_count column
                    message["token_count"] = row[5]

                messages.append(message)

            return messages

    def to_json(self) -> str:
        """
        Convert the current conversation to a JSON string.

        Returns:
            str: JSON string representation of the conversation
        """
        return json.dumps(
            self.to_dict(), indent=2, cls=DateTimeEncoder
        )

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
                json.dump(
                    self.to_dict(), f, indent=2, cls=DateTimeEncoder
                )
            return True
        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Failed to save conversation to JSON: {e}"
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
                # Convert timestamp string back to datetime if it exists
                if "timestamp" in message:
                    try:
                        datetime.datetime.fromisoformat(
                            message["timestamp"]
                        )
                    except (ValueError, TypeError):
                        message["timestamp"]

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

    def get_last_message(self) -> Optional[Dict]:
        """
        Get the last message from the current conversation.

        Returns:
            Optional[Dict]: The last message or None if conversation is empty
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT 1
            """,
                (self.current_conversation_id,),
            ).fetchone()

            if not result:
                return None

            content = result[2]  # content column
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass

            message = {
                "role": result[1],  # role column
                "content": content,
            }

            if result[3]:  # timestamp column
                message["timestamp"] = result[3]
            if result[4]:  # message_type column
                message["message_type"] = result[4]
            if result[5]:  # metadata column
                message["metadata"] = json.loads(result[5])
            if result[6]:  # token_count column
                message["token_count"] = result[6]

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
            result = conn.execute(
                f"""
                SELECT role, COUNT(*) as count
                FROM {self.table_name}
                WHERE conversation_id = ?
                GROUP BY role
            """,
                (self.current_conversation_id,),
            ).fetchall()

            return {row[0]: row[1] for row in result}

    def get_messages_by_role(self, role: str) -> List[Dict]:
        """
        Get all messages from a specific role in the current conversation.

        Args:
            role (str): Role to filter messages by

        Returns:
            List[Dict]: List of messages from the specified role
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE conversation_id = ? AND role = ?
                ORDER BY id ASC
            """,
                (self.current_conversation_id, role),
            ).fetchall()

            messages = []
            for row in result:
                content = row[2]  # content column
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                message = {
                    "role": row[1],  # role column
                    "content": content,
                }

                if row[3]:  # timestamp column
                    message["timestamp"] = row[3]
                if row[4]:  # message_type column
                    message["message_type"] = row[4]
                if row[5]:  # metadata column
                    message["metadata"] = json.loads(row[5])
                if row[6]:  # token_count column
                    message["token_count"] = row[6]

                messages.append(message)

            return messages

    def get_conversation_summary(self) -> Dict:
        """
        Get a summary of the current conversation.

        Returns:
            Dict: Summary of the conversation including message counts, roles, and time range
        """
        with self._get_connection() as conn:
            result = conn.execute(
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
            ).fetchone()

            return {
                "conversation_id": self.current_conversation_id,
                "total_messages": result[0],
                "unique_roles": result[1],
                "first_message_time": result[2],
                "last_message_time": result[3],
                "total_tokens": result[4],
                "roles": self.count_messages_by_role(),
            }

    def get_conversation_as_dict(self) -> Dict:
        """Get the entire conversation as a dictionary with messages and metadata."""
        messages = self.get_messages()
        stats = self.get_statistics()

        return {
            "conversation_id": self.current_conversation_id,
            "messages": messages,
            "metadata": {
                "total_messages": stats["total_messages"],
                "unique_roles": stats["unique_roles"],
                "total_tokens": stats["total_tokens"],
                "first_message": stats["first_message"],
                "last_message": stats["last_message"],
                "roles": self.count_messages_by_role(),
            },
        }

    def get_conversation_by_role_dict(self) -> Dict[str, List[Dict]]:
        """Get the conversation organized by roles."""
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT role, content, timestamp, message_type, metadata, token_count
                FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
            """,
                (self.current_conversation_id,),
            ).fetchall()

            role_dict = {}
            for row in result:
                role = row[0]
                content = row[1]
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                message = {
                    "content": content,
                    "timestamp": row[2],
                    "message_type": row[3],
                    "metadata": (
                        json.loads(row[4]) if row[4] else None
                    ),
                    "token_count": row[5],
                }

                if role not in role_dict:
                    role_dict[role] = []
                role_dict[role].append(message)

            return role_dict

    def get_conversation_timeline_dict(self) -> Dict[str, List[Dict]]:
        """
        Get the conversation organized by timestamps.

        Returns:
            Dict[str, List[Dict]]: Dictionary with dates as keys and lists of messages as values
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT 
                    DATE(timestamp) as date,
                    role,
                    content,
                    timestamp,
                    message_type,
                    metadata,
                    token_count
                FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """,
                (self.current_conversation_id,),
            ).fetchall()

            timeline_dict = {}
            for row in result:
                date = row[0]
                content = row[2]
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                message = {
                    "role": row[1],
                    "content": content,
                    "timestamp": row[3],
                    "message_type": row[4],
                    "metadata": (
                        json.loads(row[5]) if row[5] else None
                    ),
                    "token_count": row[6],
                }

                if date not in timeline_dict:
                    timeline_dict[date] = []
                timeline_dict[date].append(message)

            return timeline_dict

    def get_conversation_metadata_dict(self) -> Dict:
        """Get detailed metadata about the conversation."""
        with self._get_connection() as conn:
            # Get basic statistics
            stats = self.get_statistics()

            # Get message type distribution
            type_dist = conn.execute(
                f"""
                SELECT message_type, COUNT(*) as count
                FROM {self.table_name}
                WHERE conversation_id = ?
                GROUP BY message_type
            """,
                (self.current_conversation_id,),
            ).fetchall()

            # Get average tokens per message
            avg_tokens = conn.execute(
                f"""
                SELECT AVG(token_count) as avg_tokens
                FROM {self.table_name}
                WHERE conversation_id = ? AND token_count IS NOT NULL
            """,
                (self.current_conversation_id,),
            ).fetchone()

            # Get message frequency by hour
            hourly_freq = conn.execute(
                f"""
                SELECT 
                    EXTRACT(HOUR FROM timestamp) as hour,
                    COUNT(*) as count
                FROM {self.table_name}
                WHERE conversation_id = ?
                GROUP BY hour
                ORDER BY hour
            """,
                (self.current_conversation_id,),
            ).fetchall()

            return {
                "conversation_id": self.current_conversation_id,
                "basic_stats": stats,
                "message_type_distribution": {
                    row[0]: row[1] for row in type_dist if row[0]
                },
                "average_tokens_per_message": (
                    avg_tokens[0] if avg_tokens[0] is not None else 0
                ),
                "hourly_message_frequency": {
                    row[0]: row[1] for row in hourly_freq
                },
                "role_distribution": self.count_messages_by_role(),
            }

    def save_as_yaml(self, filename: str) -> bool:
        """Save the current conversation to a YAML file."""
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

    def load_from_yaml(self, filename: str) -> bool:
        """Load a conversation from a YAML file."""
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

    def delete(self, index: str):
        """Delete a message from the conversation history."""
        with self._get_connection() as conn:
            conn.execute(
                f"DELETE FROM {self.table_name} WHERE id = ? AND conversation_id = ?",
                (index, self.current_conversation_id),
            )

    def update(
        self, index: str, role: str, content: Union[str, dict]
    ):
        """Update a message in the conversation history."""
        if isinstance(content, (dict, list)):
            content = json.dumps(content)

        with self._get_connection() as conn:
            conn.execute(
                f"""
                UPDATE {self.table_name}
                SET role = ?, content = ?
                WHERE id = ? AND conversation_id = ?
                """,
                (role, content, index, self.current_conversation_id),
            )

    def query(self, index: str) -> Dict:
        """Query a message in the conversation history."""
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE id = ? AND conversation_id = ?
                """,
                (index, self.current_conversation_id),
            ).fetchone()

            if not result:
                return {}

            content = result[2]
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass

            return {
                "role": result[1],
                "content": content,
                "timestamp": result[3],
                "message_type": result[4],
                "metadata": (
                    json.loads(result[5]) if result[5] else None
                ),
                "token_count": result[6],
            }

    def search(self, keyword: str) -> List[Dict]:
        """Search for messages containing a keyword."""
        return self.search_messages(keyword)

    def display_conversation(self, detailed: bool = False):
        """Display the conversation history."""
        print(self.get_str())

    def export_conversation(self, filename: str):
        """Export the conversation history to a file."""
        self.save_as_json(filename)

    def import_conversation(self, filename: str):
        """Import a conversation history from a file."""
        self.load_from_json(filename)

    def return_history_as_string(self) -> str:
        """Return the conversation history as a string."""
        return self.get_str()

    def clear(self):
        """Clear the conversation history."""
        with self._get_connection() as conn:
            conn.execute(
                f"DELETE FROM {self.table_name} WHERE conversation_id = ?",
                (self.current_conversation_id,),
            )

    def truncate_memory_with_tokenizer(self):
        """Truncate the conversation history based on token count."""
        if not self.tokenizer:
            return

        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT id, content, token_count
                FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (self.current_conversation_id,),
            ).fetchall()

            total_tokens = 0
            ids_to_keep = []

            for row in result:
                token_count = row[2] or self.tokenizer.count_tokens(
                    row[1]
                )
                if total_tokens + token_count <= self.context_length:
                    total_tokens += token_count
                    ids_to_keep.append(row[0])
                else:
                    break

            if ids_to_keep:
                ids_str = ",".join(map(str, ids_to_keep))
                conn.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE conversation_id = ?
                    AND id NOT IN ({ids_str})
                    """,
                    (self.current_conversation_id,),
                )

    def get_visible_messages(
        self, agent: Callable, turn: int
    ) -> List[Dict]:
        """
        Get the visible messages for a given agent and turn.

        Args:
            agent (Agent): The agent.
            turn (int): The turn number.

        Returns:
            List[Dict]: The list of visible messages.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT * FROM {self.table_name}
                WHERE conversation_id = ?
                AND CAST(json_extract(metadata, '$.turn') AS INTEGER) < ?
                ORDER BY id ASC
                """,
                (self.current_conversation_id, turn),
            ).fetchall()

            visible_messages = []
            for row in result:
                metadata = json.loads(row[5]) if row[5] else {}
                visible_to = metadata.get("visible_to", "all")

                if visible_to == "all" or (
                    agent and agent.agent_name in visible_to
                ):
                    content = row[2]  # content column
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass

                    message = {
                        "role": row[1],
                        "content": content,
                        "visible_to": visible_to,
                        "turn": metadata.get("turn"),
                    }
                    visible_messages.append(message)

            return visible_messages

    def return_messages_as_list(self) -> List[str]:
        """Return the conversation messages as a list of formatted strings.

        Returns:
            list: List of messages formatted as 'role: content'.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT role, content FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (self.current_conversation_id,),
            ).fetchall()

            return [
                f"{row[0]}: {json.loads(row[1]) if isinstance(row[1], str) and row[1].startswith('{') else row[1]}"
                for row in result
            ]

    def return_messages_as_dictionary(self) -> List[Dict]:
        """Return the conversation messages as a list of dictionaries.

        Returns:
            list: List of dictionaries containing role and content of each message.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT role, content FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (self.current_conversation_id,),
            ).fetchall()

            messages = []
            for row in result:
                content = row[1]
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                messages.append(
                    {
                        "role": row[0],
                        "content": content,
                    }
                )
            return messages

    def add_tool_output_to_agent(self, role: str, tool_output: dict):
        """Add a tool output to the conversation history.

        Args:
            role (str): The role of the tool.
            tool_output (dict): The output from the tool to be added.
        """
        self.add(role, tool_output, message_type=MessageType.TOOL)

    def get_final_message(self) -> str:
        """Return the final message from the conversation history.

        Returns:
            str: The final message formatted as 'role: content'.
        """
        last_message = self.get_last_message()
        if not last_message:
            return ""
        return f"{last_message['role']}: {last_message['content']}"

    def get_final_message_content(self) -> Union[str, dict]:
        """Return the content of the final message from the conversation history.

        Returns:
            Union[str, dict]: The content of the final message.
        """
        last_message = self.get_last_message()
        if not last_message:
            return ""
        return last_message["content"]

    def return_all_except_first(self) -> List[Dict]:
        """Return all messages except the first one.

        Returns:
            list: List of messages except the first one.
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT role, content, timestamp, message_type, metadata, token_count
                FROM {self.table_name}
                WHERE conversation_id = ?
                ORDER BY id ASC
                LIMIT -1 OFFSET 2
                """,
                (self.current_conversation_id,),
            ).fetchall()

            messages = []
            for row in result:
                content = row[1]
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

                message = {
                    "role": row[0],
                    "content": content,
                }
                if row[2]:  # timestamp
                    message["timestamp"] = row[2]
                if row[3]:  # message_type
                    message["message_type"] = row[3]
                if row[4]:  # metadata
                    message["metadata"] = json.loads(row[4])
                if row[5]:  # token_count
                    message["token_count"] = row[5]

                messages.append(message)
            return messages

    def return_all_except_first_string(self) -> str:
        """Return all messages except the first one as a string.

        Returns:
            str: All messages except the first one as a string.
        """
        messages = self.return_all_except_first()
        return "\n".join(f"{msg['content']}" for msg in messages)
