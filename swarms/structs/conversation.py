import concurrent.futures
import datetime
import json
import os
import threading
import uuid
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Literal,
    Any,
)

import yaml

from swarms.structs.base_structure import BaseStructure
from swarms.utils.any_to_str import any_to_str
from swarms.utils.formatter import formatter
from swarms.utils.litellm_tokenizer import count_tokens

if TYPE_CHECKING:
    from swarms.structs.agent import Agent

from loguru import logger


def generate_conversation_id():
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())


def get_conversation_dir():
    """Get the directory for storing conversation logs."""
    # Get the current working directory
    conversation_dir = os.path.join(os.getcwd(), "conversations")
    try:
        os.makedirs(conversation_dir, mode=0o755, exist_ok=True)
    except Exception as e:
        logger.error(
            f"Failed to create conversations directory: {str(e)}"
        )
        # Fallback to the same directory as the script
        conversation_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "conversations",
        )
        os.makedirs(conversation_dir, mode=0o755, exist_ok=True)
    return conversation_dir


# Define available providers
providers = Literal["mem0", "in-memory"]


class Conversation(BaseStructure):
    """
    A class to manage a conversation history, allowing for the addition, deletion,
    and retrieval of messages, as well as saving and loading the conversation
    history in various formats.

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
        conversation_history (list): List to store the history of messages.
    """

    def __init__(
        self,
        id: str = generate_conversation_id(),
        name: str = None,
        system_prompt: Optional[str] = None,
        time_enabled: bool = False,
        autosave: bool = False,  # Changed default to False
        save_enabled: bool = False,  # New parameter to control if saving is enabled
        save_filepath: str = None,
        load_filepath: str = None,  # New parameter to specify which file to load from
        tokenizer: Callable = None,
        context_length: int = 8192,
        rules: str = None,
        custom_rules_prompt: str = None,
        user: str = "User:",
        save_as_yaml: bool = False,
        save_as_json_bool: bool = False,
        token_count: bool = True,
        provider: providers = "in-memory",
        conversations_dir: Optional[str] = None,
        message_id_on: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Initialize all attributes first
        self.id = id
        self.name = name or id
        self.system_prompt = system_prompt
        self.time_enabled = time_enabled
        self.autosave = autosave
        self.save_enabled = save_enabled
        self.conversations_dir = conversations_dir
        self.message_id_on = message_id_on

        # Handle save filepath
        if save_enabled and save_filepath:
            self.save_filepath = save_filepath
        elif save_enabled and conversations_dir:
            self.save_filepath = os.path.join(
                conversations_dir, f"{self.id}.json"
            )
        else:
            self.save_filepath = None

        self.load_filepath = load_filepath
        self.conversation_history = []
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.rules = rules
        self.custom_rules_prompt = custom_rules_prompt
        self.user = user
        self.save_as_yaml = save_as_yaml
        self.save_as_json_bool = save_as_json_bool
        self.token_count = token_count
        self.provider = provider

        # Create conversation directory if saving is enabled
        if self.save_enabled and self.conversations_dir:
            os.makedirs(self.conversations_dir, exist_ok=True)

        # Try to load existing conversation or initialize new one
        self.setup()

    def setup(self):
        """Set up the conversation by either loading existing data or initializing new."""
        if self.load_filepath and os.path.exists(self.load_filepath):
            try:
                self.load_from_json(self.load_filepath)
                logger.info(
                    f"Loaded existing conversation from {self.load_filepath}"
                )
            except Exception as e:
                logger.error(f"Failed to load conversation: {str(e)}")
                self._initialize_new_conversation()
        elif self.save_filepath and os.path.exists(
            self.save_filepath
        ):
            try:
                self.load_from_json(self.save_filepath)
                logger.info(
                    f"Loaded existing conversation from {self.save_filepath}"
                )
            except Exception as e:
                logger.error(f"Failed to load conversation: {str(e)}")
                self._initialize_new_conversation()
        else:
            self._initialize_new_conversation()

    def _initialize_new_conversation(self):
        """Initialize a new conversation with system prompt and rules."""
        if self.system_prompt is not None:
            self.add("System", self.system_prompt)

        if self.rules is not None:
            self.add(self.user or "User", self.rules)

        if self.custom_rules_prompt is not None:
            self.add(self.user or "User", self.custom_rules_prompt)

        if self.tokenizer is not None:
            self.truncate_memory_with_tokenizer()

    def _autosave(self):
        """Automatically save the conversation if autosave is enabled."""
        if self.autosave and self.save_filepath:
            try:
                self.save_as_json(self.save_filepath)
            except Exception as e:
                logger.error(
                    f"Failed to autosave conversation: {str(e)}"
                )

    def mem0_provider(self):
        try:
            from mem0 import AsyncMemory
        except ImportError:
            logger.warning(
                "mem0ai is not installed. Please install it to use the Conversation class."
            )
            return None

        try:
            memory = AsyncMemory()
            return memory
        except Exception as e:
            logger.error(
                f"Failed to initialize AsyncMemory: {str(e)}"
            )
            return None

    def add_in_memory(
        self,
        role: str,
        content: Union[str, dict, list, Any],
        *args,
        **kwargs,
    ):
        """Add a message to the conversation history.

        Args:
            role (str): The role of the speaker (e.g., 'User', 'System').
            content (Union[str, dict, list]): The content of the message to be added.
        """
        # Base message with role and timestamp
        message = {
            "role": role,
            "content": content,
        }

        if self.time_enabled:
            message["timestamp"] = datetime.datetime.now().isoformat()

        if self.message_id_on:
            message["message_id"] = str(uuid.uuid4())

        # Add message to conversation history
        self.conversation_history.append(message)

        if self.token_count is True:
            self._count_tokens(content, message)

        # Autosave after adding message, but only if saving is enabled
        if self.autosave and self.save_enabled and self.save_filepath:
            try:
                self.save_as_json(self.save_filepath)
            except Exception as e:
                logger.error(
                    f"Failed to autosave conversation: {str(e)}"
                )

    def add_mem0(
        self,
        role: str,
        content: Union[str, dict, list],
        metadata: Optional[dict] = None,
    ):
        """Add a message to the conversation history using the Mem0 provider."""
        if self.provider == "mem0":
            memory = self.mem0_provider()
            memory.add(
                messages=content,
                agent_id=role,
                run_id=self.id,
                metadata=metadata,
            )

    def add(
        self,
        role: str,
        content: Union[str, dict, list],
        metadata: Optional[dict] = None,
    ):
        """Add a message to the conversation history."""
        if self.provider == "in-memory":
            self.add_in_memory(role, content)
        elif self.provider == "mem0":
            self.add_mem0(
                role=role, content=content, metadata=metadata
            )
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def add_multiple_messages(
        self, roles: List[str], contents: List[Union[str, dict, list]]
    ):
        return self.add_multiple(roles, contents)

    def _count_tokens(self, content: str, message: dict):
        # If token counting is enabled, do it in a separate thread
        if self.token_count is True:

            # Define a function to count tokens and update the message
            def count_tokens_thread():
                tokens = count_tokens(any_to_str(content))
                # Update the message that's already in the conversation history
                message["token_count"] = int(tokens)

                # If autosave is enabled, save after token count is updated
                if self.autosave:
                    self.save_as_json(self.save_filepath)

            # Start a new thread for token counting
            token_thread = threading.Thread(
                target=count_tokens_thread
            )
            token_thread.daemon = (
                True  # Make thread terminate when main program exits
            )
            token_thread.start()

    def add_multiple(
        self,
        roles: List[str],
        contents: List[Union[str, dict, list, any]],
    ):
        """Add multiple messages to the conversation history."""
        if len(roles) != len(contents):
            raise ValueError(
                "Number of roles and contents must match."
            )

        # Now create a formula to get 25% of available cpus
        max_workers = int(os.cpu_count() * 0.25)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(self.add, role, content)
                for role, content in zip(roles, contents)
            ]
            concurrent.futures.wait(futures)

    def delete(self, index: str):
        """Delete a message from the conversation history.

        Args:
            index (str): Index of the message to delete.
        """
        self.conversation_history.pop(index)

    def update(self, index: str, role, content):
        """Update a message in the conversation history.

        Args:
            index (str): Index of the message to update.
            role (str): Role of the speaker.
            content (Union[str, dict]): New content of the message.
        """
        self.conversation_history[index] = {
            "role": role,
            "content": content,
        }

    def query(self, index: str):
        """Query a message in the conversation history.

        Args:
            index (str): Index of the message to query.

        Returns:
            dict: The message with its role and content.
        """
        return self.conversation_history[index]

    def search(self, keyword: str):
        """Search for a message in the conversation history.

        Args:
            keyword (str): Keyword to search for.

        Returns:
            list: List of messages containing the keyword.
        """
        return [
            msg
            for msg in self.conversation_history
            if keyword in msg["content"]
        ]

    def display_conversation(self, detailed: bool = False):
        """Display the conversation history.

        Args:
            detailed (bool, optional): Flag to display detailed information. Defaults to False.
        """
        for message in self.conversation_history:
            content = message["content"]
            role = message["role"]

            # Format the message content
            if isinstance(content, (dict, list)):
                content = json.dumps(content, indent=2)

            # Create the display string
            display_str = f"{role}: {content}"

            # Add details if requested
            if detailed:
                display_str += f"\nTimestamp: {message.get('timestamp', 'Unknown')}"
                display_str += f"\nMessage ID: {message.get('message_id', 'Unknown')}"
                if "token_count" in message:
                    display_str += (
                        f"\nTokens: {message['token_count']}"
                    )

            formatter.print_panel(display_str)

    def export_conversation(self, filename: str, *args, **kwargs):
        """Export the conversation history to a file.

        Args:
            filename (str): Filename to export to.
        """
        with open(filename, "w") as f:
            for message in self.conversation_history:
                f.write(f"{message['role']}: {message['content']}\n")

    def import_conversation(self, filename: str):
        """Import a conversation history from a file.

        Args:
            filename (str): Filename to import from.
        """
        with open(filename) as f:
            for line in f:
                role, content = line.split(": ", 1)
                self.add(role, content.strip())

    def count_messages_by_role(self):
        """Count the number of messages by role.

        Returns:
            dict: A dictionary with counts of messages by role.
        """
        counts = {
            "system": 0,
            "user": 0,
            "assistant": 0,
            "function": 0,
        }
        for message in self.conversation_history:
            counts[message["role"]] += 1
        return counts

    def return_history_as_string(self):
        """Return the conversation history as a string.

        Returns:
            str: The conversation history formatted as a string.
        """
        formatted_messages = []
        for message in self.conversation_history:
            formatted_messages.append(
                f"{message['role']}: {message['content']}"
            )

        return "\n\n".join(formatted_messages)

    def get_str(self) -> str:
        """Get the conversation history as a string.

        Returns:
            str: The conversation history.
        """
        return self.return_history_as_string()

    def save_as_json(self, filename: str = None):
        """Save the conversation history as a JSON file.

        Args:
            filename (str): Filename to save the conversation history.
        """
        # Don't save if saving is disabled
        if not self.save_enabled:
            return

        save_path = filename or self.save_filepath
        if save_path is not None:
            try:
                # Prepare metadata
                metadata = {
                    "id": self.id,
                    "name": self.name,
                    "created_at": datetime.datetime.now().isoformat(),
                    "system_prompt": self.system_prompt,
                    "rules": self.rules,
                    "custom_rules_prompt": self.custom_rules_prompt,
                }

                # Prepare save data
                save_data = {
                    "metadata": metadata,
                    "history": self.conversation_history,
                }

                # Create directory if it doesn't exist
                os.makedirs(
                    os.path.dirname(save_path),
                    mode=0o755,
                    exist_ok=True,
                )

                # Write directly to file
                with open(save_path, "w") as f:
                    json.dump(save_data, f, indent=2)

                # Only log explicit saves, not autosaves
                if not self.autosave:
                    logger.info(
                        f"Successfully saved conversation to {save_path}"
                    )
            except Exception as e:
                logger.error(f"Failed to save conversation: {str(e)}")

    def load_from_json(self, filename: str):
        """Load the conversation history from a JSON file.

        Args:
            filename (str): Filename to load from.
        """
        if filename is not None and os.path.exists(filename):
            try:
                with open(filename) as f:
                    data = json.load(f)

                # Load metadata
                metadata = data.get("metadata", {})
                self.id = metadata.get("id", self.id)
                self.name = metadata.get("name", self.name)
                self.system_prompt = metadata.get(
                    "system_prompt", self.system_prompt
                )
                self.rules = metadata.get("rules", self.rules)
                self.custom_rules_prompt = metadata.get(
                    "custom_rules_prompt", self.custom_rules_prompt
                )

                # Load conversation history
                self.conversation_history = data.get("history", [])

                logger.info(
                    f"Successfully loaded conversation from {filename}"
                )
            except Exception as e:
                logger.error(f"Failed to load conversation: {str(e)}")
                raise

    def search_keyword_in_conversation(self, keyword: str):
        """Search for a keyword in the conversation history.

        Args:
            keyword (str): Keyword to search for.

        Returns:
            list: List of messages containing the keyword.
        """
        return [
            msg
            for msg in self.conversation_history
            if keyword in msg["content"]
        ]

    def truncate_memory_with_tokenizer(self):
        """
        Truncates the conversation history based on the total number of tokens using a tokenizer.

        Returns:
            None
        """
        total_tokens = 0
        truncated_history = []

        for message in self.conversation_history:
            role = message.get("role")
            content = message.get("content")
            tokens = self.tokenizer.count_tokens(
                text=content
            )  # Count the number of tokens
            count = tokens  # Assign the token count
            total_tokens += count

            if total_tokens <= self.context_length:
                truncated_history.append(message)
            else:
                remaining_tokens = self.context_length - (
                    total_tokens - count
                )
                truncated_content = content[
                    :remaining_tokens
                ]  # Truncate the content based on the remaining tokens
                truncated_message = {
                    "role": role,
                    "content": truncated_content,
                }
                truncated_history.append(truncated_message)
                break

        self.conversation_history = truncated_history

    def clear(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def to_json(self):
        """Convert the conversation history to a JSON string.

        Returns:
            str: The conversation history as a JSON string.
        """
        return json.dumps(self.conversation_history, indent=4)

    def to_dict(self):
        """Convert the conversation history to a dictionary.

        Returns:
            list: The conversation history as a list of dictionaries.
        """
        return self.conversation_history

    def to_yaml(self):
        """Convert the conversation history to a YAML string.

        Returns:
            str: The conversation history as a YAML string.
        """
        return yaml.dump(self.conversation_history)

    def get_visible_messages(self, agent: "Agent", turn: int):
        """
        Get the visible messages for a given agent and turn.

        Args:
            agent (Agent): The agent.
            turn (int): The turn number.

        Returns:
            List[Dict]: The list of visible messages.
        """
        # Get the messages before the current turn
        prev_messages = [
            message
            for message in self.conversation_history
            if message["turn"] < turn
        ]

        visible_messages = []
        for message in prev_messages:
            if (
                message["visible_to"] == "all"
                or agent.agent_name in message["visible_to"]
            ):
                visible_messages.append(message)
        return visible_messages

    def get_last_message_as_string(self):
        """Fetch the last message from the conversation history.

        Returns:
            str: The last message formatted as 'role: content'.
        """
        if self.provider == "mem0":
            memory = self.mem0_provider()
            return memory.get_all(run_id=self.id)
        elif self.provider == "in-memory":
            return f"{self.conversation_history[-1]['role']}: {self.conversation_history[-1]['content']}"
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def return_messages_as_list(self):
        """Return the conversation messages as a list of formatted strings.

        Returns:
            list: List of messages formatted as 'role: content'.
        """
        return [
            f"{message['role']}: {message['content']}"
            for message in self.conversation_history
        ]

    def return_messages_as_dictionary(self):
        """Return the conversation messages as a list of dictionaries.

        Returns:
            list: List of dictionaries containing role and content of each message.
        """
        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in self.conversation_history
        ]

    def add_tool_output_to_agent(self, role: str, tool_output: dict):
        """
        Add a tool output to the conversation history.

        Args:
            role (str): The role of the tool.
            tool_output (dict): The output from the tool to be added.
        """
        self.add(role, tool_output)

    def return_json(self):
        """Return the conversation messages as a JSON string.

        Returns:
            str: The conversation messages formatted as a JSON string.
        """
        return json.dumps(
            self.return_messages_as_dictionary(), indent=4
        )

    def get_final_message(self):
        """Return the final message from the conversation history.

        Returns:
            str: The final message formatted as 'role: content'.
        """
        return f"{self.conversation_history[-1]['role']}: {self.conversation_history[-1]['content']}"

    def get_final_message_content(self):
        """Return the content of the final message from the conversation history.

        Returns:
            str: The content of the final message.
        """
        output = self.conversation_history[-1]["content"]
        # print(output)
        return output

    def return_all_except_first(self):
        """Return all messages except the first one.

        Returns:
            list: List of messages except the first one.
        """
        return self.conversation_history[2:]

    def return_all_except_first_string(self):
        """Return all messages except the first one as a string.

        Returns:
            str: All messages except the first one as a string.
        """
        return "\n".join(
            [
                f"{msg['content']}"
                for msg in self.conversation_history[2:]
            ]
        )

    def batch_add(self, messages: List[dict]):
        """Batch add messages to the conversation history.

        Args:
            messages (List[dict]): List of messages to add.
        """
        self.conversation_history.extend(messages)

    def clear_memory(self):
        """Clear the memory of the conversation."""
        self.conversation_history = []

    @classmethod
    def load_conversation(
        cls,
        name: str,
        conversations_dir: Optional[str] = None,
        load_filepath: Optional[str] = None,
    ) -> "Conversation":
        """Load a conversation from saved file by name or specific file.

        Args:
            name (str): Name of the conversation to load
            conversations_dir (Optional[str]): Directory containing conversations
            load_filepath (Optional[str]): Specific file to load from

        Returns:
            Conversation: The loaded conversation object
        """
        if load_filepath:
            return cls(
                name=name,
                load_filepath=load_filepath,
                save_enabled=False,  # Don't enable saving when loading specific file
            )

        conv_dir = conversations_dir or get_conversation_dir()
        # Try loading by name first
        filepath = os.path.join(conv_dir, f"{name}.json")

        # If not found by name, try loading by ID
        if not os.path.exists(filepath):
            filepath = os.path.join(conv_dir, f"{name}")
            if not os.path.exists(filepath):
                logger.warning(
                    f"No conversation found with name or ID: {name}"
                )
                return cls(
                    name=name,
                    conversations_dir=conv_dir,
                    save_enabled=True,
                )

        return cls(
            name=name,
            conversations_dir=conv_dir,
            load_filepath=filepath,
            save_enabled=True,
        )

    @classmethod
    def list_conversations(
        cls, conversations_dir: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """List all saved conversations.

        Args:
            conversations_dir (Optional[str]): Directory containing conversations

        Returns:
            List[Dict[str, str]]: List of conversation metadata
        """
        conv_dir = conversations_dir or get_conversation_dir()
        if not os.path.exists(conv_dir):
            return []

        conversations = []
        seen_ids = (
            set()
        )  # Track seen conversation IDs to avoid duplicates

        for filename in os.listdir(conv_dir):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(conv_dir, filename)
                    with open(filepath) as f:
                        data = json.load(f)
                        metadata = data.get("metadata", {})
                        conv_id = metadata.get("id")
                        name = metadata.get("name")
                        created_at = metadata.get("created_at")

                        # Skip if we've already seen this ID or if required fields are missing
                        if (
                            not all([conv_id, name, created_at])
                            or conv_id in seen_ids
                        ):
                            continue

                        seen_ids.add(conv_id)
                        conversations.append(
                            {
                                "id": conv_id,
                                "name": name,
                                "created_at": created_at,
                                "filepath": filepath,
                            }
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Skipping corrupted conversation file: {filename}"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Failed to read conversation {filename}: {str(e)}"
                    )
                    continue

        # Sort by creation date, newest first
        return sorted(
            conversations, key=lambda x: x["created_at"], reverse=True
        )


# # Example usage
# # conversation = Conversation()
# conversation = Conversation(token_count=True)
# conversation.add("user", "Hello, how are you?")
# conversation.add("assistant", "I am doing well, thanks.")
# # conversation.add(
# #     "assistant", {"name": "tool_1", "output": "Hello, how are you?"}
# # )
# # print(conversation.return_json())

# # # print(conversation.get_last_message_as_string())
# print(conversation.return_json())
# # # conversation.add("assistant", "I am doing well, thanks.")
# # # # print(conversation.to_json())
# # print(type(conversation.to_dict()))
# # print(conversation.to_yaml())
