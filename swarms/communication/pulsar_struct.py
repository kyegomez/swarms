import json
import yaml
import threading
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import uuid
from loguru import logger
from swarms.communication.base_communication import (
    BaseCommunication,
    Message,
    MessageType,
)


class PulsarConnectionError(Exception):
    """Exception raised for Pulsar connection errors."""

    pass


class PulsarOperationError(Exception):
    """Exception raised for Pulsar operation errors."""

    pass


class PulsarConversation(BaseCommunication):
    """
    A Pulsar-based implementation of the conversation interface.
    Uses Apache Pulsar for message storage and retrieval.

    Attributes:
        client (pulsar.Client): The Pulsar client instance
        producer (pulsar.Producer): The Pulsar producer for sending messages
        consumer (pulsar.Consumer): The Pulsar consumer for receiving messages
        topic (str): The Pulsar topic name
        subscription_name (str): The subscription name for the consumer
        conversation_id (str): Unique identifier for the conversation
        cache_enabled (bool): Flag to enable prompt caching
        cache_stats (dict): Statistics about cache usage
        cache_lock (threading.Lock): Lock for thread-safe cache operations
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
        pulsar_host: str = "pulsar://localhost:6650",
        topic: str = "conversation",
        *args,
        **kwargs,
    ):
        """Initialize the Pulsar conversation interface."""
        # Lazy load Pulsar with auto-installation
        try:
            import pulsar

            self.pulsar = pulsar
            self.pulsar_available = True
        except ImportError:
            # Auto-install pulsar-client if not available
            print(
                "📦 Pulsar client not found. Installing automatically..."
            )
            try:
                import subprocess
                import sys

                # Install pulsar-client
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "pulsar-client",
                    ]
                )
                print("✅ Pulsar client installed successfully!")

                # Try importing again
                import pulsar

                self.pulsar = pulsar
                self.pulsar_available = True
                print("✅ Pulsar loaded successfully!")

            except Exception as e:
                self.pulsar_available = False
                logger.error(
                    f"Failed to auto-install Pulsar client. Please install manually with 'pip install pulsar-client': {e}"
                )
                raise ImportError(
                    f"Failed to auto-install Pulsar client. Please install manually with 'pip install pulsar-client': {e}"
                )

        logger.info(
            f"Initializing PulsarConversation with host: {pulsar_host}"
        )

        self.conversation_id = str(uuid.uuid4())
        self.topic = f"{topic}-{self.conversation_id}"
        self.subscription_name = f"sub-{self.conversation_id}"

        try:
            # Initialize Pulsar client and producer/consumer
            logger.debug(
                f"Connecting to Pulsar broker at {pulsar_host}"
            )
            self.client = pulsar.Client(pulsar_host)

            logger.debug(f"Creating producer for topic: {self.topic}")
            self.producer = self.client.create_producer(self.topic)

            logger.debug(
                f"Creating consumer with subscription: {self.subscription_name}"
            )
            self.consumer = self.client.subscribe(
                self.topic, self.subscription_name
            )
            logger.info("Successfully connected to Pulsar broker")

        except pulsar.ConnectError as e:
            error_msg = f"Failed to connect to Pulsar broker at {pulsar_host}: {str(e)}"
            logger.error(error_msg)
            raise PulsarConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error while initializing Pulsar connection: {str(e)}"
            logger.error(error_msg)
            raise PulsarOperationError(error_msg)

        # Store configuration
        self.system_prompt = system_prompt
        self.time_enabled = time_enabled
        self.autosave = autosave
        self.save_filepath = save_filepath
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.rules = rules
        self.custom_rules_prompt = custom_rules_prompt
        self.user = user
        self.auto_save = auto_save
        self.save_as_yaml = save_as_yaml
        self.save_as_json_bool = save_as_json_bool
        self.token_count = token_count

        # Cache configuration
        self.cache_enabled = cache_enabled
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
        self.cache_lock = threading.Lock()

        # Add system prompt if provided
        if system_prompt:
            logger.debug("Adding system prompt to conversation")
            self.add("system", system_prompt, MessageType.SYSTEM)

        # Add rules if provided
        if rules:
            logger.debug("Adding rules to conversation")
            self.add("system", rules, MessageType.SYSTEM)

        # Add custom rules prompt if provided
        if custom_rules_prompt:
            logger.debug("Adding custom rules prompt to conversation")
            self.add(user, custom_rules_prompt, MessageType.USER)

        logger.info(
            f"PulsarConversation initialized with ID: {self.conversation_id}"
        )

    def add(
        self,
        role: str,
        content: Union[str, dict, list],
        message_type: Optional[MessageType] = None,
        metadata: Optional[Dict] = None,
        token_count: Optional[int] = None,
    ) -> int:
        """Add a message to the conversation."""
        try:
            message = {
                "id": str(uuid.uuid4()),
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "message_type": (
                    message_type.value if message_type else None
                ),
                "metadata": metadata or {},
                "token_count": token_count,
                "conversation_id": self.conversation_id,
            }

            logger.debug(
                f"Adding message with ID {message['id']} from role: {role}"
            )

            # Send message to Pulsar
            message_data = json.dumps(message).encode("utf-8")
            self.producer.send(message_data)

            logger.debug(
                f"Successfully added message with ID: {message['id']}"
            )
            return message["id"]

        except pulsar.ConnectError as e:
            error_msg = f"Failed to send message to Pulsar: Connection error: {str(e)}"
            logger.error(error_msg)
            raise PulsarConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Failed to add message: {str(e)}"
            logger.error(error_msg)
            raise PulsarOperationError(error_msg)

    def batch_add(self, messages: List[Message]) -> List[int]:
        """Add multiple messages to the conversation."""
        message_ids = []
        for message in messages:
            msg_id = self.add(
                message.role,
                message.content,
                message.message_type,
                message.metadata,
                message.token_count,
            )
            message_ids.append(msg_id)
        return message_ids

    def get_messages(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict]:
        """Get messages with optional pagination."""
        messages = []
        try:
            logger.debug("Retrieving messages from Pulsar")
            while True:
                try:
                    msg = self.consumer.receive(timeout_millis=1000)
                    messages.append(json.loads(msg.data()))
                    self.consumer.acknowledge(msg)
                except pulsar.Timeout:
                    break  # No more messages available
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
                    continue

            logger.debug(f"Retrieved {len(messages)} messages")

            if offset is not None:
                messages = messages[offset:]
            if limit is not None:
                messages = messages[:limit]

            return messages

        except pulsar.ConnectError as e:
            error_msg = f"Failed to receive messages from Pulsar: Connection error: {str(e)}"
            logger.error(error_msg)
            raise PulsarConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Failed to get messages: {str(e)}"
            logger.error(error_msg)
            raise PulsarOperationError(error_msg)

    def delete(self, message_id: str):
        """Delete a message from the conversation."""
        # In Pulsar, messages cannot be deleted individually
        # We would need to implement a soft delete by marking messages
        pass

    def update(
        self, message_id: str, role: str, content: Union[str, dict]
    ):
        """Update a message in the conversation."""
        # In Pulsar, messages are immutable
        # We would need to implement updates as new messages with update metadata
        new_message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "updates": message_id,
            "conversation_id": self.conversation_id,
        }
        self.producer.send(json.dumps(new_message).encode("utf-8"))

    def query(self, message_id: str) -> Dict:
        """Query a message in the conversation."""
        messages = self.get_messages()
        for message in messages:
            if message["id"] == message_id:
                return message
        return None

    def search(self, keyword: str) -> List[Dict]:
        """Search for messages containing a keyword."""
        messages = self.get_messages()
        return [
            msg for msg in messages if keyword in str(msg["content"])
        ]

    def get_str(self) -> str:
        """Get the conversation history as a string."""
        messages = self.get_messages()
        return "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in messages]
        )

    def display_conversation(self, detailed: bool = False):
        """Display the conversation history."""
        messages = self.get_messages()
        for msg in messages:
            if detailed:
                print(f"ID: {msg['id']}")
                print(f"Role: {msg['role']}")
                print(f"Content: {msg['content']}")
                print(f"Timestamp: {msg['timestamp']}")
                print("---")
            else:
                print(f"{msg['role']}: {msg['content']}")

    def export_conversation(self, filename: str):
        """Export the conversation history to a file."""
        messages = self.get_messages()
        with open(filename, "w") as f:
            json.dump(messages, f, indent=2)

    def import_conversation(self, filename: str):
        """Import a conversation history from a file."""
        with open(filename, "r") as f:
            messages = json.load(f)
        for msg in messages:
            self.add(
                msg["role"],
                msg["content"],
                (
                    MessageType(msg["message_type"])
                    if msg.get("message_type")
                    else None
                ),
                msg.get("metadata"),
                msg.get("token_count"),
            )

    def count_messages_by_role(self) -> Dict[str, int]:
        """Count messages by role."""
        messages = self.get_messages()
        counts = {}
        for msg in messages:
            role = msg["role"]
            counts[role] = counts.get(role, 0) + 1
        return counts

    def return_history_as_string(self) -> str:
        """Return the conversation history as a string."""
        return self.get_str()

    def clear(self):
        """Clear the conversation history."""
        try:
            logger.info(
                f"Clearing conversation with ID: {self.conversation_id}"
            )

            # Close existing producer and consumer
            if hasattr(self, "consumer"):
                self.consumer.close()
            if hasattr(self, "producer"):
                self.producer.close()

            # Create new conversation ID and topic
            self.conversation_id = str(uuid.uuid4())
            self.topic = f"conversation-{self.conversation_id}"
            self.subscription_name = f"sub-{self.conversation_id}"

            # Recreate producer and consumer
            logger.debug(
                f"Creating new producer for topic: {self.topic}"
            )
            self.producer = self.client.create_producer(self.topic)

            logger.debug(
                f"Creating new consumer with subscription: {self.subscription_name}"
            )
            self.consumer = self.client.subscribe(
                self.topic, self.subscription_name
            )

            logger.info(
                f"Successfully cleared conversation. New ID: {self.conversation_id}"
            )

        except pulsar.ConnectError as e:
            error_msg = f"Failed to clear conversation: Connection error: {str(e)}"
            logger.error(error_msg)
            raise PulsarConnectionError(error_msg)
        except Exception as e:
            error_msg = f"Failed to clear conversation: {str(e)}"
            logger.error(error_msg)
            raise PulsarOperationError(error_msg)

    def to_dict(self) -> List[Dict]:
        """Convert the conversation history to a dictionary."""
        return self.get_messages()

    def to_json(self) -> str:
        """Convert the conversation history to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_yaml(self) -> str:
        """Convert the conversation history to a YAML string."""
        return yaml.dump(self.to_dict())

    def save_as_json(self, filename: str):
        """Save the conversation history as a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_json(self, filename: str):
        """Load the conversation history from a JSON file."""
        self.import_conversation(filename)

    def save_as_yaml(self, filename: str):
        """Save the conversation history as a YAML file."""
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    def load_from_yaml(self, filename: str):
        """Load the conversation history from a YAML file."""
        with open(filename, "r") as f:
            messages = yaml.safe_load(f)
        for msg in messages:
            self.add(
                msg["role"],
                msg["content"],
                (
                    MessageType(msg["message_type"])
                    if msg.get("message_type")
                    else None
                ),
                msg.get("metadata"),
                msg.get("token_count"),
            )

    def get_last_message(self) -> Optional[Dict]:
        """Get the last message from the conversation history."""
        messages = self.get_messages()
        return messages[-1] if messages else None

    def get_last_message_as_string(self) -> str:
        """Get the last message as a formatted string."""
        last_message = self.get_last_message()
        if last_message:
            return (
                f"{last_message['role']}: {last_message['content']}"
            )
        return ""

    def get_messages_by_role(self, role: str) -> List[Dict]:
        """Get all messages from a specific role."""
        messages = self.get_messages()
        return [msg for msg in messages if msg["role"] == role]

    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation."""
        messages = self.get_messages()
        return {
            "conversation_id": self.conversation_id,
            "message_count": len(messages),
            "roles": list(set(msg["role"] for msg in messages)),
            "start_time": (
                messages[0]["timestamp"] if messages else None
            ),
            "end_time": (
                messages[-1]["timestamp"] if messages else None
            ),
        }

    def get_statistics(self) -> Dict:
        """Get statistics about the conversation."""
        messages = self.get_messages()
        return {
            "total_messages": len(messages),
            "messages_by_role": self.count_messages_by_role(),
            "cache_stats": self.get_cache_stats(),
        }

    def get_conversation_id(self) -> str:
        """Get the current conversation ID."""
        return self.conversation_id

    def start_new_conversation(self) -> str:
        """Start a new conversation and return its ID."""
        self.clear()
        return self.conversation_id

    def delete_current_conversation(self) -> bool:
        """Delete the current conversation."""
        self.clear()
        return True

    def search_messages(self, query: str) -> List[Dict]:
        """Search for messages containing specific text."""
        return self.search(query)

    def update_message(
        self,
        message_id: int,
        content: Union[str, dict, list],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Update an existing message."""
        message = self.query(message_id)
        if message:
            self.update(message_id, message["role"], content)
            return True
        return False

    def get_conversation_metadata_dict(self) -> Dict:
        """Get detailed metadata about the conversation."""
        return self.get_conversation_summary()

    def get_conversation_timeline_dict(self) -> Dict[str, List[Dict]]:
        """Get the conversation organized by timestamps."""
        messages = self.get_messages()
        timeline = {}
        for msg in messages:
            date = msg["timestamp"].split("T")[0]
            if date not in timeline:
                timeline[date] = []
            timeline[date].append(msg)
        return timeline

    def get_conversation_by_role_dict(self) -> Dict[str, List[Dict]]:
        """Get the conversation organized by roles."""
        messages = self.get_messages()
        by_role = {}
        for msg in messages:
            role = msg["role"]
            if role not in by_role:
                by_role[role] = []
            by_role[role].append(msg)
        return by_role

    def get_conversation_as_dict(self) -> Dict:
        """Get the entire conversation as a dictionary with messages and metadata."""
        return {
            "metadata": self.get_conversation_metadata_dict(),
            "messages": self.get_messages(),
            "statistics": self.get_statistics(),
        }

    def truncate_memory_with_tokenizer(self):
        """Truncate the conversation history based on token count."""
        if not self.tokenizer:
            return

        messages = self.get_messages()
        total_tokens = 0
        truncated_messages = []

        for msg in messages:
            content = msg["content"]
            tokens = self.tokenizer.count_tokens(str(content))

            if total_tokens + tokens <= self.context_length:
                truncated_messages.append(msg)
                total_tokens += tokens
            else:
                break

        # Clear and re-add truncated messages
        self.clear()
        for msg in truncated_messages:
            self.add(
                msg["role"],
                msg["content"],
                (
                    MessageType(msg["message_type"])
                    if msg.get("message_type")
                    else None
                ),
                msg.get("metadata"),
                msg.get("token_count"),
            )

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage."""
        with self.cache_lock:
            return {
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "cached_tokens": self.cache_stats["cached_tokens"],
                "total_tokens": self.cache_stats["total_tokens"],
                "hit_rate": (
                    self.cache_stats["hits"]
                    / (
                        self.cache_stats["hits"]
                        + self.cache_stats["misses"]
                    )
                    if (
                        self.cache_stats["hits"]
                        + self.cache_stats["misses"]
                    )
                    > 0
                    else 0
                ),
            }

    def __del__(self):
        """Cleanup Pulsar resources."""
        try:
            logger.debug("Cleaning up Pulsar resources")
            if hasattr(self, "consumer"):
                self.consumer.close()
            if hasattr(self, "producer"):
                self.producer.close()
            if hasattr(self, "client"):
                self.client.close()
            logger.info("Successfully cleaned up Pulsar resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    @classmethod
    def check_pulsar_availability(
        cls, pulsar_host: str = "pulsar://localhost:6650"
    ) -> bool:
        """
        Check if Pulsar is available and accessible.

        Args:
            pulsar_host (str): The Pulsar host to check

        Returns:
            bool: True if Pulsar is available and accessible, False otherwise
        """
        try:
            import pulsar

        except ImportError:
            logger.error("Pulsar client library is not installed")
            return False

        try:
            logger.debug(
                f"Checking Pulsar availability at {pulsar_host}"
            )
            client = pulsar.Client(pulsar_host)
            client.close()
            logger.info("Pulsar is available and accessible")
            return True
        except Exception as e:
            logger.error(f"Pulsar is not accessible: {str(e)}")
            return False

    def health_check(self) -> Dict[str, bool]:
        """
        Perform a health check of the Pulsar connection and components.

        Returns:
            Dict[str, bool]: Health status of different components
        """
        health = {
            "client_connected": False,
            "producer_active": False,
            "consumer_active": False,
        }

        try:
            # Check client
            if hasattr(self, "client"):
                health["client_connected"] = True

            # Check producer
            if hasattr(self, "producer"):
                # Try to send a test message
                test_msg = json.dumps(
                    {"type": "health_check"}
                ).encode("utf-8")
                self.producer.send(test_msg)
                health["producer_active"] = True

            # Check consumer
            if hasattr(self, "consumer"):
                try:
                    msg = self.consumer.receive(timeout_millis=1000)
                    self.consumer.acknowledge(msg)
                    health["consumer_active"] = True
                except pulsar.Timeout:
                    pass

            logger.info(f"Health check results: {health}")
            return health

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return health
