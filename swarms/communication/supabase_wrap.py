import datetime
import json
import logging
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from swarms.communication.base_communication import (
    BaseCommunication,
    Message,
    MessageType,
)

# Try to import loguru logger, fallback to standard logging
try:
    from loguru import logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    logger = None


# Custom Exceptions for Supabase Communication
class SupabaseConnectionError(Exception):
    """Custom exception for Supabase connection errors."""

    pass


class SupabaseOperationError(Exception):
    """Custom exception for Supabase operation errors."""

    pass


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


class SupabaseConversation(BaseCommunication):
    """
    A Supabase-backed implementation of the BaseCommunication class for managing
    conversation history using a Supabase (PostgreSQL) database.

    Prerequisites:
    - supabase-py library: pip install supabase
    - Valid Supabase project URL and API key
    - Network access to your Supabase instance

    Attributes:
        supabase_url (str): URL of the Supabase project.
        supabase_key (str): Anon or service key for the Supabase project.
        client (supabase.Client): The Supabase client instance.
        table_name (str): Name of the table in Supabase to store conversations.
        current_conversation_id (Optional[str]): ID of the currently active conversation.
        tokenizer (Any): Tokenizer for counting tokens in messages.
        context_length (int): Maximum number of tokens for context window.
        time_enabled (bool): Flag to prepend timestamps to messages.
        enable_logging (bool): Flag to enable logging.
        logger (logging.Logger | loguru.Logger): Logger instance.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        system_prompt: Optional[str] = None,
        time_enabled: bool = False,
        autosave: bool = False,  # Standardized parameter name - less relevant for DB-backed, but kept for interface
        save_filepath: str = None,  # Used for export/import
        tokenizer: Any = None,
        context_length: int = 8192,
        rules: str = None,
        custom_rules_prompt: str = None,
        user: str = "User:",
        save_as_yaml: bool = True,  # Default export format
        save_as_json_bool: bool = False,  # Alternative export format
        token_count: bool = True,
        cache_enabled: bool = True,  # Currently for token counting
        table_name: str = "conversations",
        enable_timestamps: bool = True,  # DB schema handles this with DEFAULT NOW()
        enable_logging: bool = True,
        use_loguru: bool = True,
        max_retries: int = 3,  # For Supabase API calls (not implemented yet, supabase-py might handle)
        *args,
        **kwargs,
    ):
        # Lazy load Supabase with auto-installation
        try:
            from supabase import Client, create_client

            self.supabase_client = Client
            self.create_client = create_client
            self.supabase_available = True
        except ImportError:
            # Auto-install supabase if not available
            print(
                "📦 Supabase not found. Installing automatically..."
            )
            try:
                import subprocess
                import sys

                # Install supabase
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "supabase",
                    ]
                )
                print("✅ Supabase installed successfully!")

                # Try importing again
                from supabase import Client, create_client

                self.supabase_client = Client
                self.create_client = create_client
                self.supabase_available = True
                print("✅ Supabase loaded successfully!")

            except Exception as e:
                self.supabase_available = False
                if logger:
                    logger.error(
                        f"Failed to auto-install Supabase. Please install manually with 'pip install supabase': {e}"
                    )
                raise ImportError(
                    f"Failed to auto-install Supabase. Please install manually with 'pip install supabase': {e}"
                )

        # Store initialization parameters - BaseCommunication.__init__ is just pass
        self.system_prompt = system_prompt
        self.time_enabled = time_enabled
        self.autosave = autosave
        self.save_filepath = save_filepath
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.rules = rules
        self.custom_rules_prompt = custom_rules_prompt
        self.user = user
        self.save_as_yaml_on_export = save_as_yaml
        self.save_as_json_on_export = save_as_json_bool
        self.calculate_token_count = token_count
        self.cache_enabled = cache_enabled

        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.table_name = table_name
        self.enable_timestamps = (
            enable_timestamps  # DB handles actual timestamping
        )
        self.enable_logging = enable_logging
        self.use_loguru = use_loguru and LOGURU_AVAILABLE
        self.max_retries = max_retries

        # Setup logging
        if self.enable_logging:
            if self.use_loguru and logger:
                self.logger = logger
            else:
                self.logger = logging.getLogger(__name__)
                if not self.logger.handlers:
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                    handler.setFormatter(formatter)
                    self.logger.addHandler(handler)
                    self.logger.setLevel(logging.INFO)
        else:
            # Create a null logger that does nothing
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())

        self.current_conversation_id: Optional[str] = None
        self._lock = (
            threading.Lock()
        )  # For thread-safe operations if any (e.g. token calculation)

        try:
            self.client = self.create_client(
                supabase_url, supabase_key
            )
            if self.enable_logging:
                self.logger.info(
                    f"Successfully initialized Supabase client for URL: {supabase_url}"
                )
        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Failed to initialize Supabase client: {e}"
                )
            raise SupabaseConnectionError(
                f"Failed to connect to Supabase: {e}"
            )

        self._init_db()  # Verifies table existence
        self.start_new_conversation()  # Initializes a conversation ID

        # Add initial prompts if provided
        if self.system_prompt:
            self.add(
                role="system",
                content=self.system_prompt,
                message_type=MessageType.SYSTEM,
            )
        if self.rules:
            # Assuming rules are spoken by the system or user based on context
            self.add(
                role="system",
                content=self.rules,
                message_type=MessageType.SYSTEM,
            )
        if self.custom_rules_prompt:
            self.add(
                role=self.user,
                content=self.custom_rules_prompt,
                message_type=MessageType.USER,
            )

    def _init_db(self):
        """
        Initialize the database and create necessary tables.
        Creates the table if it doesn't exist, similar to SQLite implementation.
        """
        # First, try to create the table if it doesn't exist
        try:
            # Use Supabase RPC to execute raw SQL for table creation
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id BIGSERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                message_type TEXT,
                metadata JSONB,
                token_count INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """

            # Try to create index as well

            # Attempt to create table using RPC function
            # Note: This requires a stored procedure to be created in Supabase
            # If RPC is not available, we'll fall back to checking if table exists
            try:
                # Try using a custom RPC function if available
                self.client.rpc(
                    "exec_sql", {"sql": create_table_sql}
                ).execute()
                if self.enable_logging:
                    self.logger.info(
                        f"Successfully created or verified table '{self.table_name}' using RPC."
                    )
            except Exception as rpc_error:
                if self.enable_logging:
                    self.logger.debug(
                        f"RPC table creation failed (expected if no custom function): {rpc_error}"
                    )

                # Fallback: Try to verify table exists, if not provide helpful error
                try:
                    response = (
                        self.client.table(self.table_name)
                        .select("id")
                        .limit(1)
                        .execute()
                    )
                    if (
                        response.error
                        and "does not exist"
                        in str(response.error).lower()
                    ):
                        # Table doesn't exist, try alternative creation method
                        self._create_table_fallback()
                    elif response.error:
                        raise SupabaseOperationError(
                            f"Error accessing table: {response.error.message}"
                        )
                    else:
                        if self.enable_logging:
                            self.logger.info(
                                f"Successfully verified existing table '{self.table_name}'."
                            )
                except Exception as table_check_error:
                    if (
                        "does not exist"
                        in str(table_check_error).lower()
                        or "relation"
                        in str(table_check_error).lower()
                    ):
                        # Table definitely doesn't exist, provide creation instructions
                        self._handle_missing_table()
                    else:
                        raise SupabaseOperationError(
                            f"Failed to access or create table: {table_check_error}"
                        )

        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Database initialization failed: {e}"
                )
            raise SupabaseOperationError(
                f"Failed to initialize database: {e}"
            )

    def _create_table_fallback(self):
        """
        Fallback method to create table when RPC is not available.
        Attempts to use Supabase's admin API or provides clear instructions.
        """
        try:
            # Try using the admin API if available (requires service role key)
            # This might work if the user is using a service role key
            admin_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id BIGSERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                message_type TEXT,
                metadata JSONB,
                token_count INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_conversation_id 
            ON {self.table_name} (conversation_id);
            """

            # Note: This might not work with all Supabase configurations
            # but we attempt it anyway
            if hasattr(self.client, "postgrest") and hasattr(
                self.client.postgrest, "rpc"
            ):
                self.client.postgrest.rpc(
                    "exec_sql", {"query": admin_sql}
                ).execute()
                if self.enable_logging:
                    self.logger.info(
                        f"Successfully created table '{self.table_name}' using admin API."
                    )
                return
        except Exception as e:
            if self.enable_logging:
                self.logger.debug(
                    f"Admin API table creation failed: {e}"
                )

        # If all else fails, call the missing table handler
        self._handle_missing_table()

    def _handle_missing_table(self):
        """
        Handle the case where the table doesn't exist and can't be created automatically.
        Provides clear instructions for manual table creation.
        """
        table_creation_sql = f"""
-- Run this SQL in your Supabase SQL Editor to create the required table:

CREATE TABLE IF NOT EXISTS {self.table_name} (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    message_type TEXT,
    metadata JSONB,
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for better query performance:
CREATE INDEX IF NOT EXISTS idx_{self.table_name}_conversation_id 
ON {self.table_name} (conversation_id);

-- Optional: Enable Row Level Security (RLS) for production:
ALTER TABLE {self.table_name} ENABLE ROW LEVEL SECURITY;

-- Optional: Create RLS policy (customize according to your needs):
CREATE POLICY "Users can manage their own conversations" ON {self.table_name}
    FOR ALL USING (true);  -- Adjust this policy based on your security requirements
"""

        error_msg = (
            f"Table '{self.table_name}' does not exist in your Supabase database and cannot be created automatically. "
            f"Please create it manually by running the following SQL in your Supabase SQL Editor:\n\n{table_creation_sql}\n\n"
            f"Alternatively, you can create a custom RPC function in Supabase to enable automatic table creation. "
            f"Visit your Supabase dashboard > SQL Editor and create this function:\n\n"
            f"CREATE OR REPLACE FUNCTION exec_sql(sql TEXT)\n"
            f"RETURNS TEXT AS $$\n"
            f"BEGIN\n"
            f"    EXECUTE sql;\n"
            f"    RETURN 'SUCCESS';\n"
            f"END;\n"
            f"$$ LANGUAGE plpgsql SECURITY DEFINER;\n\n"
            f"After creating either the table or the RPC function, retry initializing the SupabaseConversation."
        )

        if self.enable_logging:
            self.logger.error(error_msg)
        raise SupabaseOperationError(error_msg)

    def _handle_api_response(
        self, response, operation_name: str = "Supabase operation"
    ):
        """Handles Supabase API response, checking for errors and returning data."""
        # The new supabase-py client structure: response has .data and .count attributes
        # Errors are raised as exceptions rather than being in response.error
        try:
            if hasattr(response, "data"):
                # Return the data, which could be None, a list, or a dict
                return response.data
            else:
                # Fallback for older response structures or direct data
                return response
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"{operation_name} failed: {e}")
            raise SupabaseOperationError(
                f"{operation_name} failed: {e}"
            )

    def _serialize_content(
        self, content: Union[str, dict, list]
    ) -> str:
        """Serializes content to JSON string if it's a dict or list."""
        if isinstance(content, (dict, list)):
            return json.dumps(content, cls=DateTimeEncoder)
        return str(content)

    def _deserialize_content(
        self, content_str: str
    ) -> Union[str, dict, list]:
        """Deserializes content from JSON string if it looks like JSON. More robust approach."""
        if not content_str:
            return content_str

        # Always try to parse as JSON first, fall back to string
        try:
            return json.loads(content_str)
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON, return as string
            return content_str

    def _serialize_metadata(
        self, metadata: Optional[Dict]
    ) -> Optional[str]:
        """Serializes metadata dict to JSON string using simplified encoder."""
        if metadata is None:
            return None
        try:
            return json.dumps(
                metadata, default=str, ensure_ascii=False
            )
        except (TypeError, ValueError) as e:
            if self.enable_logging:
                self.logger.warning(
                    f"Failed to serialize metadata: {e}"
                )
            return None

    def _deserialize_metadata(
        self, metadata_str: Optional[str]
    ) -> Optional[Dict]:
        """Deserializes metadata from JSON string with better error handling."""
        if metadata_str is None:
            return None
        try:
            return json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError) as e:
            if self.enable_logging:
                self.logger.warning(
                    f"Failed to deserialize metadata: {metadata_str[:50]}... Error: {e}"
                )
            return None

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID using UUID and timestamp."""
        timestamp = datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y%m%d_%H%M%S_%f")
        unique_id = str(uuid.uuid4())[:8]
        return f"conv_{timestamp}_{unique_id}"

    def start_new_conversation(self) -> str:
        """Starts a new conversation and returns its ID."""
        self.current_conversation_id = (
            self._generate_conversation_id()
        )
        self.logger.info(
            f"Started new conversation with ID: {self.current_conversation_id}"
        )
        return self.current_conversation_id

    def add(
        self,
        role: str,
        content: Union[str, dict, list],
        message_type: Optional[MessageType] = None,
        metadata: Optional[Dict] = None,
        token_count: Optional[int] = None,
    ) -> int:
        """Add a message to the current conversation history in Supabase."""
        if self.current_conversation_id is None:
            self.start_new_conversation()

        serialized_content = self._serialize_content(content)
        current_timestamp_iso = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        message_data = {
            "conversation_id": self.current_conversation_id,
            "role": role,
            "content": serialized_content,
            "timestamp": current_timestamp_iso,  # Supabase will use its default if not provided / column allows NULL
            "message_type": (
                message_type.value if message_type else None
            ),
            "metadata": self._serialize_metadata(metadata),
            # token_count handled below
        }

        # Calculate token_count if enabled and not provided
        if (
            self.calculate_token_count
            and token_count is None
            and self.tokenizer
        ):
            try:
                # For now, do this synchronously. For long content, consider async/threading.
                message_data["token_count"] = (
                    self.tokenizer.count_tokens(str(content))
                )
            except Exception as e:
                if self.enable_logging:
                    self.logger.warning(
                        f"Failed to count tokens for content: {e}"
                    )
        elif token_count is not None:
            message_data["token_count"] = token_count

        # Filter out None values to let Supabase handle defaults or NULLs appropriately
        message_to_insert = {
            k: v for k, v in message_data.items() if v is not None
        }

        try:
            response = (
                self.client.table(self.table_name)
                .insert(message_to_insert)
                .execute()
            )
            data = self._handle_api_response(response, "add_message")
            if data and len(data) > 0 and "id" in data[0]:
                inserted_id = data[0]["id"]
                if self.enable_logging:
                    self.logger.debug(
                        f"Added message with ID {inserted_id} to conversation {self.current_conversation_id}"
                    )
                return inserted_id
            if self.enable_logging:
                self.logger.error(
                    f"Failed to retrieve ID for inserted message in conversation {self.current_conversation_id}"
                )
            raise SupabaseOperationError(
                "Failed to retrieve ID for inserted message."
            )
        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Error adding message to Supabase: {e}"
                )
            raise SupabaseOperationError(f"Error adding message: {e}")

    def batch_add(self, messages: List[Message]) -> List[int]:
        """Add multiple messages to the current conversation history in Supabase."""
        if self.current_conversation_id is None:
            self.start_new_conversation()

        messages_to_insert = []
        for msg_obj in messages:
            serialized_content = self._serialize_content(
                msg_obj.content
            )
            current_timestamp_iso = (
                msg_obj.timestamp
                or datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
            )

            msg_data = {
                "conversation_id": self.current_conversation_id,
                "role": msg_obj.role,
                "content": serialized_content,
                "timestamp": current_timestamp_iso,
                "message_type": (
                    msg_obj.message_type.value
                    if msg_obj.message_type
                    else None
                ),
                "metadata": self._serialize_metadata(
                    msg_obj.metadata
                ),
            }

            # Token count
            current_token_count = msg_obj.token_count
            if (
                self.calculate_token_count
                and current_token_count is None
                and self.tokenizer
            ):
                try:
                    current_token_count = self.tokenizer.count_tokens(
                        str(msg_obj.content)
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to count tokens for batch message: {e}"
                    )
            if current_token_count is not None:
                msg_data["token_count"] = current_token_count

            messages_to_insert.append(
                {k: v for k, v in msg_data.items() if v is not None}
            )

        if not messages_to_insert:
            return []

        try:
            response = (
                self.client.table(self.table_name)
                .insert(messages_to_insert)
                .execute()
            )
            data = self._handle_api_response(
                response, "batch_add_messages"
            )
            inserted_ids = [
                item["id"] for item in data if "id" in item
            ]
            if len(inserted_ids) != len(messages_to_insert):
                self.logger.warning(
                    "Mismatch in expected and inserted message counts during batch_add."
                )
            self.logger.debug(
                f"Batch added {len(inserted_ids)} messages to conversation {self.current_conversation_id}"
            )
            return inserted_ids
        except Exception as e:
            self.logger.error(
                f"Error batch adding messages to Supabase: {e}"
            )
            raise SupabaseOperationError(
                f"Error batch adding messages: {e}"
            )

    def _format_row_to_dict(self, row: Dict) -> Dict:
        """Helper to format a raw row from Supabase to our standard message dict."""
        formatted_message = {
            "id": row.get("id"),
            "role": row.get("role"),
            "content": self._deserialize_content(
                row.get("content", "")
            ),
            "timestamp": row.get("timestamp"),
            "message_type": row.get("message_type"),
            "metadata": self._deserialize_metadata(
                row.get("metadata")
            ),
            "token_count": row.get("token_count"),
            "conversation_id": row.get("conversation_id"),
            "created_at": row.get("created_at"),
        }
        # Clean None values from the root, but keep them within deserialized content/metadata
        return {
            k: v
            for k, v in formatted_message.items()
            if v is not None
            or k in ["metadata", "token_count", "message_type"]
        }

    def get_messages(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict]:
        """Get messages from the current conversation with optional pagination."""
        if self.current_conversation_id is None:
            return []
        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("conversation_id", self.current_conversation_id)
                .order("timestamp", desc=False)
            )  # Assuming 'timestamp' or 'id' for ordering

            if limit is not None:
                query = query.limit(limit)
            if offset is not None:
                query = query.offset(offset)

            response = query.execute()
            data = self._handle_api_response(response, "get_messages")
            return [self._format_row_to_dict(row) for row in data]
        except Exception as e:
            self.logger.error(
                f"Error getting messages from Supabase: {e}"
            )
            raise SupabaseOperationError(
                f"Error getting messages: {e}"
            )

    def get_str(self) -> str:
        """Get the current conversation history as a formatted string."""
        messages_dict = self.get_messages()
        conv_str = []
        for msg in messages_dict:
            ts_prefix = (
                f"[{msg['timestamp']}] "
                if msg.get("timestamp") and self.time_enabled
                else ""
            )
            # Content might be dict/list if deserialized
            content_display = msg["content"]
            if isinstance(content_display, (dict, list)):
                content_display = json.dumps(
                    content_display, indent=2, cls=DateTimeEncoder
                )
            conv_str.append(
                f"{ts_prefix}{msg['role']}: {content_display}"
            )
        return "\n".join(conv_str)

    def display_conversation(self, detailed: bool = False):
        """Display the conversation history."""
        # `detailed` flag might be used for more verbose printing if needed
        print(self.get_str())

    def delete(self, index: str):
        """Delete a message from the conversation history by its primary key 'id'."""
        if self.current_conversation_id is None:
            if self.enable_logging:
                self.logger.warning(
                    "Cannot delete message: No current conversation."
                )
            return

        try:
            # Handle both string and int message IDs
            try:
                message_id = int(index)
            except ValueError:
                if self.enable_logging:
                    self.logger.error(
                        f"Invalid message ID for delete: {index}. Must be an integer."
                    )
                raise ValueError(
                    f"Invalid message ID for delete: {index}. Must be an integer."
                )

            response = (
                self.client.table(self.table_name)
                .delete()
                .eq("id", message_id)
                .eq("conversation_id", self.current_conversation_id)
                .execute()
            )
            self._handle_api_response(
                response, f"delete_message (id: {message_id})"
            )
            if self.enable_logging:
                self.logger.info(
                    f"Deleted message with ID {message_id} from conversation {self.current_conversation_id}"
                )
        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Error deleting message ID {index} from Supabase: {e}"
                )
            raise SupabaseOperationError(
                f"Error deleting message ID {index}: {e}"
            )

    def update(
        self, index: str, role: str, content: Union[str, dict]
    ):
        """Update a message in the conversation history. Matches BaseCommunication signature exactly."""
        # Use the flexible internal method
        return self._update_flexible(
            index=index, role=role, content=content
        )

    def _update_flexible(
        self,
        index: Union[str, int],
        role: Optional[str] = None,
        content: Optional[Union[str, dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Internal flexible update method. Returns True if successful, False otherwise."""
        if self.current_conversation_id is None:
            if self.enable_logging:
                self.logger.warning(
                    "Cannot update message: No current conversation."
                )
            return False

        # Handle both string and int message IDs
        try:
            if isinstance(index, str):
                message_id = int(index)
            else:
                message_id = index
        except ValueError:
            if self.enable_logging:
                self.logger.error(
                    f"Invalid message ID for update: {index}. Must be an integer."
                )
            return False

        update_data = {}
        if role is not None:
            update_data["role"] = role
        if content is not None:
            update_data["content"] = self._serialize_content(content)
            if self.calculate_token_count and self.tokenizer:
                try:
                    update_data["token_count"] = (
                        self.tokenizer.count_tokens(str(content))
                    )
                except Exception as e:
                    if self.enable_logging:
                        self.logger.warning(
                            f"Failed to count tokens for updated content: {e}"
                        )
        if (
            metadata is not None
        ):  # Allows setting metadata to null by passing {} then serializing
            update_data["metadata"] = self._serialize_metadata(
                metadata
            )

        if not update_data:
            if self.enable_logging:
                self.logger.info(
                    "No fields provided to update for message."
                )
            return False

        try:
            response = (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("id", message_id)
                .eq("conversation_id", self.current_conversation_id)
                .execute()
            )

            data = self._handle_api_response(
                response, f"update_message (id: {message_id})"
            )

            # Check if any rows were actually updated
            if data and len(data) > 0:
                if self.enable_logging:
                    self.logger.info(
                        f"Updated message with ID {message_id} in conversation {self.current_conversation_id}"
                    )
                return True
            else:
                if self.enable_logging:
                    self.logger.warning(
                        f"No message found with ID {message_id} in conversation {self.current_conversation_id}"
                    )
                return False

        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Error updating message ID {message_id} in Supabase: {e}"
                )
            return False

    def query(self, index: str) -> Dict:
        """Query a message in the conversation history by its primary key 'id'. Returns empty dict if not found to match BaseCommunication signature."""
        if self.current_conversation_id is None:
            return {}
        try:
            # Handle both string and int message IDs
            try:
                message_id = int(index)
            except ValueError:
                if self.enable_logging:
                    self.logger.warning(
                        f"Invalid message ID for query: {index}. Must be an integer."
                    )
                return {}

            response = (
                self.client.table(self.table_name)
                .select("*")
                .eq("id", message_id)
                .eq("conversation_id", self.current_conversation_id)
                .maybe_single()
                .execute()
            )  # maybe_single returns one record or None

            data = self._handle_api_response(
                response, f"query_message (id: {message_id})"
            )
            if data:
                return self._format_row_to_dict(data)
            return {}
        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Error querying message ID {index} from Supabase: {e}"
                )
            return {}

    def query_optional(self, index: str) -> Optional[Dict]:
        """Query a message and return None if not found. More precise return type."""
        result = self.query(index)
        return result if result else None

    def search(self, keyword: str) -> List[Dict]:
        """Search for messages containing a keyword in their content."""
        if self.current_conversation_id is None:
            return []
        try:
            # PostgREST ilike is case-insensitive
            response = (
                self.client.table(self.table_name)
                .select("*")
                .eq("conversation_id", self.current_conversation_id)
                .ilike("content", f"%{keyword}%")
                .order("timestamp", desc=False)
                .execute()
            )
            data = self._handle_api_response(
                response, f"search_messages (keyword: {keyword})"
            )
            return [self._format_row_to_dict(row) for row in data]
        except Exception as e:
            self.logger.error(
                f"Error searching messages in Supabase: {e}"
            )
            raise SupabaseOperationError(
                f"Error searching messages: {e}"
            )

    def _export_to_file(self, filename: str, format_type: str):
        """Helper to export conversation to JSON or YAML file."""
        if self.current_conversation_id is None:
            self.logger.warning("No current conversation to export.")
            return

        data_to_export = (
            self.to_dict()
        )  # Gets messages for current_conversation_id
        try:
            with open(filename, "w") as f:
                if format_type == "json":
                    json.dump(
                        data_to_export,
                        f,
                        indent=2,
                        cls=DateTimeEncoder,
                    )
                elif format_type == "yaml":
                    yaml.dump(data_to_export, f, sort_keys=False)
                else:
                    raise ValueError(
                        f"Unsupported export format: {format_type}"
                    )
            self.logger.info(
                f"Conversation {self.current_conversation_id} exported to {filename} as {format_type}."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to export conversation to {format_type}: {e}"
            )
            raise

    def export_conversation(self, filename: str):
        """Export the current conversation history to a file (JSON or YAML based on init flags)."""
        if self.save_as_json_on_export:
            self._export_to_file(filename, "json")
        elif self.save_as_yaml_on_export:  # Default if json is false
            self._export_to_file(filename, "yaml")
        else:  # Fallback if somehow both are false
            self._export_to_file(filename, "yaml")

    def _import_from_file(self, filename: str, format_type: str):
        """Helper to import conversation from JSON or YAML file."""
        try:
            with open(filename, "r") as f:
                if format_type == "json":
                    imported_data = json.load(f)
                elif format_type == "yaml":
                    imported_data = yaml.safe_load(f)
                else:
                    raise ValueError(
                        f"Unsupported import format: {format_type}"
                    )

            if not isinstance(imported_data, list):
                raise ValueError(
                    "Imported data must be a list of messages."
                )

            # Start a new conversation for the imported data
            self.start_new_conversation()

            messages_to_batch = []
            for msg_data in imported_data:
                # Adapt to Message dataclass structure if possible
                role = msg_data.get("role")
                content = msg_data.get("content")
                if role is None or content is None:
                    self.logger.warning(
                        f"Skipping message due to missing role/content: {msg_data}"
                    )
                    continue

                messages_to_batch.append(
                    Message(
                        role=role,
                        content=content,
                        timestamp=msg_data.get(
                            "timestamp"
                        ),  # Will be handled by batch_add
                        message_type=(
                            MessageType(msg_data["message_type"])
                            if msg_data.get("message_type")
                            else None
                        ),
                        metadata=msg_data.get("metadata"),
                        token_count=msg_data.get("token_count"),
                    )
                )

            if messages_to_batch:
                self.batch_add(messages_to_batch)
            self.logger.info(
                f"Conversation imported from {filename} ({format_type}) into new ID {self.current_conversation_id}."
            )

        except Exception as e:
            self.logger.error(
                f"Failed to import conversation from {format_type}: {e}"
            )
            raise

    def import_conversation(self, filename: str):
        """Import a conversation history from a file (tries JSON then YAML)."""
        try:
            if filename.lower().endswith(".json"):
                self._import_from_file(filename, "json")
            elif filename.lower().endswith((".yaml", ".yml")):
                self._import_from_file(filename, "yaml")
            else:
                # Try JSON first, then YAML as a fallback
                try:
                    self._import_from_file(filename, "json")
                except (
                    json.JSONDecodeError,
                    ValueError,
                ):  # ValueError if not list
                    self.logger.info(
                        f"Failed to import {filename} as JSON, trying YAML."
                    )
                    self._import_from_file(filename, "yaml")
        except Exception as e:  # Catch errors from _import_from_file
            raise SupabaseOperationError(
                f"Could not import {filename}: {e}"
            )

    def count_messages_by_role(self) -> Dict[str, int]:
        """Count messages by role for the current conversation."""
        if self.current_conversation_id is None:
            return {}
        try:
            # Supabase rpc might be better for direct count, but select + python count is also fine
            # For direct DB count: self.client.rpc('count_roles', {'conv_id': self.current_conversation_id}).execute()
            messages = (
                self.get_messages()
            )  # Fetches for current_conversation_id
            counts = {}
            for msg in messages:
                role = msg.get("role", "unknown")
                counts[role] = counts.get(role, 0) + 1
            return counts
        except Exception as e:
            self.logger.error(f"Error counting messages by role: {e}")
            raise SupabaseOperationError(
                f"Error counting messages by role: {e}"
            )

    def return_history_as_string(self) -> str:
        """Return the conversation history as a string."""
        return self.get_str()

    def clear(self):
        """Clear the current conversation history from Supabase."""
        if self.current_conversation_id is None:
            self.logger.info("No current conversation to clear.")
            return
        try:
            response = (
                self.client.table(self.table_name)
                .delete()
                .eq("conversation_id", self.current_conversation_id)
                .execute()
            )
            # response.data will be a list of deleted items.
            # response.count might be available for delete operations in some supabase-py versions or configurations.
            # For now, we assume success if no error.
            self._handle_api_response(
                response,
                f"clear_conversation (id: {self.current_conversation_id})",
            )
            self.logger.info(
                f"Cleared conversation with ID: {self.current_conversation_id}"
            )
        except Exception as e:
            self.logger.error(
                f"Error clearing conversation {self.current_conversation_id} from Supabase: {e}"
            )
            raise SupabaseOperationError(
                f"Error clearing conversation: {e}"
            )

    def to_dict(self) -> List[Dict]:
        """Convert the current conversation history to a list of dictionaries."""
        return (
            self.get_messages()
        )  # Already fetches for current_conversation_id

    def to_json(self) -> str:
        """Convert the current conversation history to a JSON string."""
        return json.dumps(
            self.to_dict(), indent=2, cls=DateTimeEncoder
        )

    def to_yaml(self) -> str:
        """Convert the current conversation history to a YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False)

    def save_as_json(self, filename: str):
        """Save the current conversation history as a JSON file."""
        self._export_to_file(filename, "json")

    def load_from_json(self, filename: str):
        """Load a conversation history from a JSON file into a new conversation."""
        self._import_from_file(filename, "json")

    def save_as_yaml(self, filename: str):
        """Save the current conversation history as a YAML file."""
        self._export_to_file(filename, "yaml")

    def load_from_yaml(self, filename: str):
        """Load a conversation history from a YAML file into a new conversation."""
        self._import_from_file(filename, "yaml")

    def get_last_message(self) -> Optional[Dict]:
        """Get the last message from the current conversation history."""
        if self.current_conversation_id is None:
            return None
        try:
            response = (
                self.client.table(self.table_name)
                .select("*")
                .eq("conversation_id", self.current_conversation_id)
                .order("timestamp", desc=True)
                .limit(1)
                .maybe_single()
                .execute()
            )
            data = self._handle_api_response(
                response, "get_last_message"
            )
            return self._format_row_to_dict(data) if data else None
        except Exception as e:
            self.logger.error(
                f"Error getting last message from Supabase: {e}"
            )
            raise SupabaseOperationError(
                f"Error getting last message: {e}"
            )

    def get_last_message_as_string(self) -> str:
        """Get the last message as a formatted string."""
        last_msg = self.get_last_message()
        if not last_msg:
            return ""
        ts_prefix = (
            f"[{last_msg['timestamp']}] "
            if last_msg.get("timestamp") and self.time_enabled
            else ""
        )
        content_display = last_msg["content"]
        if isinstance(content_display, (dict, list)):
            content_display = json.dumps(
                content_display, cls=DateTimeEncoder
            )
        return f"{ts_prefix}{last_msg['role']}: {content_display}"

    def get_messages_by_role(self, role: str) -> List[Dict]:
        """Get all messages from a specific role in the current conversation."""
        if self.current_conversation_id is None:
            return []
        try:
            response = (
                self.client.table(self.table_name)
                .select("*")
                .eq("conversation_id", self.current_conversation_id)
                .eq("role", role)
                .order("timestamp", desc=False)
                .execute()
            )
            data = self._handle_api_response(
                response, f"get_messages_by_role (role: {role})"
            )
            return [self._format_row_to_dict(row) for row in data]
        except Exception as e:
            self.logger.error(
                f"Error getting messages by role '{role}' from Supabase: {e}"
            )
            raise SupabaseOperationError(
                f"Error getting messages by role '{role}': {e}"
            )

    def get_conversation_summary(self) -> Dict:
        """Get a summary of the current conversation."""
        if self.current_conversation_id is None:
            return {"error": "No current conversation."}

        # This could be optimized with an RPC call in Supabase for better performance
        # Example RPC: CREATE OR REPLACE FUNCTION get_conversation_summary(conv_id TEXT) ...
        messages = self.get_messages()
        if not messages:
            return {
                "conversation_id": self.current_conversation_id,
                "total_messages": 0,
                "unique_roles": 0,
                "first_message_time": None,
                "last_message_time": None,
                "total_tokens": 0,
                "roles": {},
            }

        roles_counts = {}
        total_tokens_sum = 0
        for msg in messages:
            roles_counts[msg["role"]] = (
                roles_counts.get(msg["role"], 0) + 1
            )
            if msg.get("token_count") is not None:
                total_tokens_sum += int(msg["token_count"])

        return {
            "conversation_id": self.current_conversation_id,
            "total_messages": len(messages),
            "unique_roles": len(roles_counts),
            "first_message_time": messages[0].get("timestamp"),
            "last_message_time": messages[-1].get("timestamp"),
            "total_tokens": total_tokens_sum,
            "roles": roles_counts,
        }

    def get_statistics(self) -> Dict:
        """Get statistics about the current conversation (alias for get_conversation_summary)."""
        return self.get_conversation_summary()

    def get_conversation_id(self) -> str:
        """Get the current conversation ID."""
        return self.current_conversation_id or ""

    def delete_current_conversation(self) -> bool:
        """Delete the current conversation. Returns True if successful."""
        if self.current_conversation_id:
            self.clear()  # clear messages for current_conversation_id
            self.logger.info(
                f"Deleted current conversation: {self.current_conversation_id}"
            )
            self.current_conversation_id = (
                None  # No active conversation after deletion
            )
            return True
        self.logger.info("No current conversation to delete.")
        return False

    def search_messages(self, query: str) -> List[Dict]:
        """Search for messages containing specific text (alias for search)."""
        return self.search(keyword=query)

    def get_conversation_metadata_dict(self) -> Dict:
        """Get detailed metadata about the conversation."""
        # Similar to get_conversation_summary, could be expanded with more DB-side aggregations if needed via RPC.
        # For now, returning the summary.
        if self.current_conversation_id is None:
            return {"error": "No current conversation."}
        summary = self.get_conversation_summary()

        # Example of additional metadata one might compute client-side or via RPC
        # message_type_distribution, average_tokens_per_message, hourly_message_frequency
        return {
            "conversation_id": self.current_conversation_id,
            "basic_stats": summary,
            # Placeholder for more detailed stats if implemented
        }

    def get_conversation_timeline_dict(self) -> Dict[str, List[Dict]]:
        """Get the conversation organized by timestamps (dates as keys)."""
        if self.current_conversation_id is None:
            return {}

        messages = (
            self.get_messages()
        )  # Assumes messages are ordered by timestamp
        timeline_dict = {}
        for msg in messages:
            try:
                # Ensure timestamp is a string and valid ISO format
                ts_str = msg.get("timestamp")
                if isinstance(ts_str, str):
                    date_key = datetime.datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")
                    if date_key not in timeline_dict:
                        timeline_dict[date_key] = []
                    timeline_dict[date_key].append(msg)
                else:
                    self.logger.warning(
                        f"Message ID {msg.get('id')} has invalid timestamp format: {ts_str}"
                    )
            except ValueError as e:
                self.logger.warning(
                    f"Could not parse timestamp for message ID {msg.get('id')}: {ts_str}, Error: {e}"
                )

        return timeline_dict

    def get_conversation_by_role_dict(self) -> Dict[str, List[Dict]]:
        """Get the conversation organized by roles."""
        if self.current_conversation_id is None:
            return {}

        messages = self.get_messages()
        role_dict = {}
        for msg in messages:
            role = msg.get("role", "unknown")
            if role not in role_dict:
                role_dict[role] = []
            role_dict[role].append(msg)
        return role_dict

    def get_conversation_as_dict(self) -> Dict:
        """Get the entire current conversation as a dictionary with messages and metadata."""
        if self.current_conversation_id is None:
            return {"error": "No current conversation."}

        return {
            "conversation_id": self.current_conversation_id,
            "messages": self.get_messages(),
            "metadata": self.get_conversation_summary(),  # Using summary as metadata
        }

    def truncate_memory_with_tokenizer(self):
        """Truncate the conversation history based on token count if a tokenizer is provided. Optimized for better performance."""
        if not self.tokenizer or self.current_conversation_id is None:
            if self.enable_logging:
                self.logger.info(
                    "Tokenizer not available or no current conversation, skipping truncation."
                )
            return

        try:
            # Fetch messages with only necessary fields for efficiency
            response = (
                self.client.table(self.table_name)
                .select("id, content, token_count")
                .eq("conversation_id", self.current_conversation_id)
                .order("timestamp", desc=False)
                .execute()
            )

            messages = self._handle_api_response(
                response, "fetch_messages_for_truncation"
            )
            if not messages:
                return

            # Calculate tokens and determine which messages to delete
            total_tokens = 0
            message_tokens = []

            for msg in messages:
                token_count = msg.get("token_count")
                if token_count is None and self.calculate_token_count:
                    # Recalculate if missing
                    content = self._deserialize_content(
                        msg.get("content", "")
                    )
                    token_count = self.tokenizer.count_tokens(
                        str(content)
                    )

                message_tokens.append(
                    {"id": msg["id"], "tokens": token_count or 0}
                )
                total_tokens += token_count or 0

            tokens_to_remove = total_tokens - self.context_length
            if tokens_to_remove <= 0:
                return  # No truncation needed

            # Collect IDs to delete (oldest first)
            ids_to_delete = []
            for msg_info in message_tokens:
                if tokens_to_remove <= 0:
                    break
                ids_to_delete.append(msg_info["id"])
                tokens_to_remove -= msg_info["tokens"]

            if not ids_to_delete:
                return

            # Batch delete for better performance
            if len(ids_to_delete) == 1:
                # Single delete
                response = (
                    self.client.table(self.table_name)
                    .delete()
                    .eq("id", ids_to_delete[0])
                    .eq(
                        "conversation_id",
                        self.current_conversation_id,
                    )
                    .execute()
                )
            else:
                # Batch delete using 'in' operator
                response = (
                    self.client.table(self.table_name)
                    .delete()
                    .in_("id", ids_to_delete)
                    .eq(
                        "conversation_id",
                        self.current_conversation_id,
                    )
                    .execute()
                )

            self._handle_api_response(
                response, "truncate_conversation_batch_delete"
            )

            if self.enable_logging:
                self.logger.info(
                    f"Truncated conversation {self.current_conversation_id}, removed {len(ids_to_delete)} oldest messages."
                )

        except Exception as e:
            if self.enable_logging:
                self.logger.error(
                    f"Error during memory truncation for conversation {self.current_conversation_id}: {e}"
                )
            # Don't re-raise, truncation is best-effort

    # Methods from duckdb_wrap.py that seem generally useful and can be adapted
    def get_visible_messages(
        self,
        agent: Optional[Callable] = None,
        turn: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get visible messages, optionally filtered by agent visibility and turn.
        Assumes 'metadata' field can contain 'visible_to' (list of agent names or 'all')
        and 'turn' (integer).
        """
        if self.current_conversation_id is None:
            return []

        # Base query
        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("conversation_id", self.current_conversation_id)
            .order("timestamp", desc=False)
        )

        # Execute and then filter in Python, as JSONB querying for array containment or
        # numeric comparison within JSON can be complex with supabase-py's fluent API.
        # For complex filtering, an RPC function in Supabase would be more efficient.

        try:
            response = query.execute()
            all_messages = self._handle_api_response(
                response, "get_visible_messages_fetch_all"
            )
        except Exception as e:
            self.logger.error(
                f"Error fetching messages for visibility check: {e}"
            )
            return []

        visible_messages = []
        for row_data in all_messages:
            msg = self._format_row_to_dict(row_data)
            metadata = (
                msg.get("metadata")
                if isinstance(msg.get("metadata"), dict)
                else {}
            )

            # Turn filtering
            if turn is not None:
                msg_turn = metadata.get("turn")
                if not (
                    isinstance(msg_turn, int) and msg_turn < turn
                ):
                    continue  # Skip if turn condition not met

            # Agent visibility filtering
            if agent is not None:
                visible_to = metadata.get("visible_to")
                agent_name_attr = getattr(
                    agent, "agent_name", None
                )  # Safely get agent_name
                if (
                    agent_name_attr is None
                ):  # If agent has no name, assume it can't see restricted msgs
                    if visible_to is not None and visible_to != "all":
                        continue
                elif (
                    isinstance(visible_to, list)
                    and agent_name_attr not in visible_to
                ):
                    continue  # Skip if agent not in visible_to list
                elif (
                    isinstance(visible_to, str)
                    and visible_to != "all"
                ):
                    # If visible_to is a string but not "all", and doesn't match agent_name
                    if visible_to != agent_name_attr:
                        continue

            visible_messages.append(msg)
        return visible_messages

    def return_messages_as_list(self) -> List[str]:
        """Return the conversation messages as a list of formatted strings 'role: content'."""
        messages_dict = self.get_messages()
        return [
            f"{msg.get('role', 'unknown')}: {self._serialize_content(msg.get('content', ''))}"
            for msg in messages_dict
        ]

    def return_messages_as_dictionary(self) -> List[Dict]:
        """Return the conversation messages as a list of dictionaries [{role: R, content: C}]."""
        messages_dict = self.get_messages()
        return [
            {
                "role": msg.get("role"),
                "content": msg.get("content"),
            }  # Content already deserialized by _format_row_to_dict
            for msg in messages_dict
        ]

    def add_tool_output_to_agent(
        self, role: str, tool_output: dict
    ):  # role is usually "tool"
        """Add a tool output to the conversation history."""
        # Assuming tool_output is a dict that should be stored as content
        self.add(
            role=role,
            content=tool_output,
            message_type=MessageType.TOOL,
        )

    def get_final_message(self) -> Optional[str]:
        """Return the final message from the conversation history as 'role: content' string."""
        last_msg = self.get_last_message()
        if not last_msg:
            return None
        content_display = last_msg["content"]
        if isinstance(content_display, (dict, list)):
            content_display = json.dumps(
                content_display, cls=DateTimeEncoder
            )
        return f"{last_msg.get('role', 'unknown')}: {content_display}"

    def get_final_message_content(
        self,
    ) -> Union[str, dict, list, None]:
        """Return the content of the final message from the conversation history."""
        last_msg = self.get_last_message()
        return last_msg.get("content") if last_msg else None

    def return_all_except_first(self) -> List[Dict]:
        """Return all messages except the first one."""
        # The limit=-1, offset=2 from duckdb_wrap is specific to its ID generation.
        # For Supabase, we fetch all and skip the first one in Python.
        all_messages = self.get_messages()
        return all_messages[1:] if len(all_messages) > 1 else []

    def return_all_except_first_string(self) -> str:
        """Return all messages except the first one as a concatenated string."""
        messages_to_format = self.return_all_except_first()
        conv_str = []
        for msg in messages_to_format:
            ts_prefix = (
                f"[{msg['timestamp']}] "
                if msg.get("timestamp") and self.time_enabled
                else ""
            )
            content_display = msg["content"]
            if isinstance(content_display, (dict, list)):
                content_display = json.dumps(
                    content_display, indent=2, cls=DateTimeEncoder
                )
            conv_str.append(
                f"{ts_prefix}{msg['role']}: {content_display}"
            )
        return "\n".join(conv_str)

    def update_message(
        self,
        message_id: int,
        content: Union[str, dict, list],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Update an existing message. Matches BaseCommunication.update_message signature exactly."""
        # Use the flexible internal method
        return self._update_flexible(
            index=message_id, content=content, metadata=metadata
        )
