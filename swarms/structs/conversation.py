import concurrent.futures
import datetime
import json
import os
import traceback
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import yaml

from swarms.utils.any_to_str import any_to_str
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


class Conversation:
    """
    A class to manage a conversation history, allowing for the addition, deletion,
    and retrieval of messages, as well as saving and loading the conversation
    history in various formats.

    The Conversation class uses in-memory storage for conversation history.

    Attributes:
        system_prompt (Optional[str]): The system prompt for the conversation.
        time_enabled (bool): Flag to enable time tracking for messages.
        autosave (bool): Flag to enable automatic saving of conversation history.
        save_filepath (str): File path for saving the conversation history.
        context_length (int): Maximum number of tokens allowed in the conversation history.
        rules (str): Rules for the conversation.
        custom_rules_prompt (str): Custom prompt for rules.
        user (str): The user identifier for messages.
        auto_save (bool): Flag to enable auto-saving of conversation history.
        save_as_yaml_on (bool): Flag to save conversation history as YAML.
        save_as_json_bool (bool): Flag to save conversation history as JSON.
        token_count (bool): Flag to enable token counting for messages.
        conversation_history (list): List to store the history of messages.
    """

    def __init__(
        self,
        id: str = generate_conversation_id(),
        name: str = "conversation-test",
        system_prompt: Optional[str] = None,
        time_enabled: bool = False,
        autosave: bool = False,
        save_filepath: str = None,
        load_filepath: str = None,
        context_length: int = 8192,
        rules: str = None,
        custom_rules_prompt: str = None,
        user: str = "User",
        save_as_yaml_on: bool = False,
        save_as_json_bool: bool = False,
        token_count: bool = False,
        message_id_on: bool = False,
        tokenizer_model_name: str = "gpt-4.1",
        conversations_dir: Optional[str] = None,
        export_method: str = "json",
        dynamic_context_window: bool = True,
        caching: bool = True,
        output_metadata: bool = False,
    ):

        # Initialize all attributes first
        self.id = id
        self.name = name
        self.save_filepath = save_filepath
        self.system_prompt = system_prompt
        self.time_enabled = time_enabled
        self.autosave = autosave
        self.conversations_dir = conversations_dir
        self.tokenizer_model_name = tokenizer_model_name
        self.message_id_on = message_id_on
        self.load_filepath = load_filepath
        self.context_length = context_length
        self.rules = rules
        self.custom_rules_prompt = custom_rules_prompt
        self.user = user
        self.save_as_yaml_on = save_as_yaml_on
        self.save_as_json_bool = save_as_json_bool
        self.token_count = token_count
        self.export_method = export_method
        self.dynamic_context_window = dynamic_context_window
        self.caching = caching
        self.output_metadata = output_metadata

        if self.name is None:
            self.name = id

        self.conversation_history = []

        self.setup_file_path()
        self.setup()

    def setup_file_path(self):
        """Set up the file path for saving the conversation and load existing data if available."""
        # Validate export method
        if self.export_method not in ["json", "yaml"]:
            raise ValueError(
                f"Invalid export_method: {self.export_method}. Must be 'json' or 'yaml'"
            )

        # Set default save filepath if not provided
        if not self.save_filepath:
            # Ensure extension matches export method
            extension = (
                ".json" if self.export_method == "json" else ".yaml"
            )
            self.save_filepath = (
                f"conversation_{self.name}{extension}"
            )
            logger.debug(
                f"Setting default save filepath to: {self.save_filepath}"
            )
        else:
            # Validate that provided filepath extension matches export method
            file_ext = os.path.splitext(self.save_filepath)[1].lower()
            expected_ext = (
                ".json" if self.export_method == "json" else ".yaml"
            )
            if file_ext != expected_ext:
                logger.warning(
                    f"Save filepath extension ({file_ext}) does not match export_method ({self.export_method}). "
                    f"Updating filepath extension to match export method."
                )
                base_name = os.path.splitext(self.save_filepath)[0]
                self.save_filepath = f"{base_name}{expected_ext}"

        self.created_at = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        # Check if file exists and load it
        if os.path.exists(self.save_filepath):
            logger.debug(
                f"Found existing conversation file at: {self.save_filepath}"
            )
            try:
                self.load(self.save_filepath)
                logger.info(
                    f"Loaded existing conversation from {self.save_filepath}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load existing conversation from {self.save_filepath}: {str(e)}"
                )
                # Keep the empty conversation_history initialized in __init__

        else:
            logger.debug(
                f"No existing conversation file found at: {self.save_filepath}"
            )

    def setup(self):
        # Set up conversations directory
        self.conversations_dir = (
            self.conversations_dir
            or os.path.join(
                os.path.expanduser("~"), ".swarms", "conversations"
            )
        )
        os.makedirs(self.conversations_dir, exist_ok=True)

        # Try to load existing conversation if it exists
        conversation_file = os.path.join(
            self.conversations_dir, f"{self.name}.json"
        )
        if os.path.exists(conversation_file):
            with open(conversation_file, "r") as f:
                saved_data = json.load(f)
                # Update attributes from saved data
                for key, value in saved_data.get(
                    "metadata", {}
                ).items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                self.conversation_history = saved_data.get(
                    "history", []
                )
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

        # if self.tokenizer is not None:
        #     self.truncate_memory_with_tokenizer()

    def _autosave(self):
        """Automatically save the conversation if autosave is enabled."""
        return self.export()

    def add_in_memory(
        self,
        role: str,
        content: Union[str, dict, list, Any],
        category: Optional[str] = None,
    ):
        """Add a message to the conversation history.

        Args:
            role (str): The role of the speaker (e.g., 'User', 'System').
            content (Union[str, dict, list]): The content of the message to be added.
            category (Optional[str]): Optional category for the message.
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

        if category:
            message["category"] = category

        # Add message to conversation history
        self.conversation_history.append(message)

        # Handle token counting in a separate thread if enabled
        if self.token_count is True:
            tokens = count_tokens(
                text=any_to_str(content),
                model=self.tokenizer_model_name,
            )
            message["token_count"] = tokens

        return message

    def export_and_count_categories(
        self,
    ) -> Dict[str, int]:
        """Export all messages with category 'input' and 'output' and count their tokens.

        This method searches through the conversation history and:
        1. Extracts all messages marked with category 'input' or 'output'
        2. Concatenates the content of each category
        3. Counts tokens for each category using the specified tokenizer model

        Args:
            tokenizer_model_name (str): Name of the model to use for tokenization

        Returns:
            Dict[str, int]: A dictionary containing:
                - input_tokens: Number of tokens in input messages
                - output_tokens: Number of tokens in output messages
                - total_tokens: Total tokens across both categories
        """
        try:
            # Extract input and output messages
            input_messages = []
            output_messages = []

            for message in self.conversation_history:
                # Get message content and ensure it's a string
                content = message.get("content", "")
                if not isinstance(content, str):
                    content = str(content)

                # Sort messages by category
                category = message.get("category", "")
                if category == "input":
                    input_messages.append(content)
                elif category == "output":
                    output_messages.append(content)

            # Join messages with spaces
            all_input_text = " ".join(input_messages)
            all_output_text = " ".join(output_messages)

            print(all_input_text)
            print(all_output_text)

            # Count tokens only if there is text
            input_tokens = (
                count_tokens(
                    all_input_text, self.tokenizer_model_name
                )
                if all_input_text.strip()
                else 0
            )
            output_tokens = (
                count_tokens(
                    all_output_text, self.tokenizer_model_name
                )
                if all_output_text.strip()
                else 0
            )
            total_tokens = input_tokens + output_tokens

            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }

        except Exception as e:
            logger.error(
                f"Error in export_and_count_categories: {str(e)}"
            )
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    def add(
        self,
        role: str,
        content: Union[str, dict, list, Any],
        metadata: Optional[dict] = None,
        category: Optional[str] = None,
    ):
        """Add a message to the conversation history.

        Args:
            role (str): The role of the speaker (e.g., 'User', 'System').
            content (Union[str, dict, list]): The content of the message to be added.
            metadata (Optional[dict]): Optional metadata for the message.
            category (Optional[str]): Optional category for the message.
        """
        result = self.add_in_memory(
            role=role, content=content, category=category
        )

        # Ensure autosave happens after the message is added
        if self.autosave:
            self._autosave()

        return result

    def add_multiple_messages(
        self, roles: List[str], contents: List[Union[str, dict, list]]
    ):
        added = self.add_multiple(roles, contents)

        if self.autosave:
            self._autosave()

        return added

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
        """Delete a message from the conversation history."""
        self.conversation_history.pop(int(index))

    def update(self, index: str, role, content):
        """Update a message in the conversation history.

        Args:
            index (int): The index of the message to update.
            role (str): The role of the speaker.
            content: The new content of the message.
        """
        if 0 <= int(index) < len(self.conversation_history):
            self.conversation_history[int(index)]["role"] = role
            self.conversation_history[int(index)]["content"] = content
        else:
            logger.warning(f"Invalid index: {index}")

    def query(self, index: str):
        """Query a message from the conversation history.

        Args:
            index (int): The index of the message to query.

        Returns:
            dict: The message at the specified index.
        """
        if 0 <= int(index) < len(self.conversation_history):
            return self.conversation_history[int(index)]
        return None

    def search(self, keyword: str):
        """Search for messages containing a keyword.

        Args:
            keyword (str): The keyword to search for.

        Returns:
            list: A list of messages containing the keyword.
        """
        return [
            message
            for message in self.conversation_history
            if keyword in str(message["content"])
        ]

    def export_conversation(self, filename: str, *args, **kwargs):
        """Export the conversation history to a file.

        Args:
            filename (str): Filename to export to.
        """
        # If the filename ends with .json, use save_as_json
        if filename.endswith(".json"):
            self.save_as_json(force=True)
        else:
            # Simple text export for non-JSON files
            with open(filename, "w", encoding="utf-8") as f:
                for message in self.conversation_history:
                    f.write(
                        f"{message['role']}: {message['content']}\n"
                    )

    def import_conversation(self, filename: str):
        """Import a conversation history from a file.

        Args:
            filename (str): Filename to import from.
        """
        self.load_from_json(filename)

    def count_messages_by_role(self):
        """Count the number of messages by role.

        Returns:
            dict: A dictionary with counts of messages by role.
        """
        # Initialize counts with expected roles
        counts = {
            "system": 0,
            "user": 0,
            "assistant": 0,
            "function": 0,
        }

        # Count messages by role
        for message in self.conversation_history:
            role = message["role"]
            if role in counts:
                counts[role] += 1
            else:
                # Handle unexpected roles dynamically
                counts[role] = counts.get(role, 0) + 1

        return counts

    def return_history_as_string(self):
        """Return the conversation history as a string.

        Returns:
            str: The conversation history formatted as a string.
        """
        if self.dynamic_context_window is True:
            return self.dynamic_auto_chunking()
        else:
            return self._return_history_as_string_worker()

    def _return_history_as_string_worker(self):
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

    def to_dict(self) -> Dict[Any, Any]:
        """
        Converts all attributes of the class into a dictionary, including all __init__ parameters
        and conversation history. Automatically extracts parameters from __init__ signature.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - metadata: All initialization parameters and their current values
                - conversation_history: The list of conversation messages
        """
        return self.conversation_history

    def save_as_json(self, force: bool = True):
        """Save the conversation history and metadata to a JSON file.

        Args:
            force (bool, optional): If True, saves regardless of autosave setting. Defaults to True.
        """
        try:
            # Check if saving is allowed
            if not self.autosave and not force:
                logger.warning(
                    "Autosave is disabled. To save anyway, call save_as_json(force=True) "
                    "or enable autosave by setting autosave=True when creating the Conversation."
                )
                return

            # Ensure we have a valid save path
            if not self.save_filepath:
                self.save_filepath = os.path.join(
                    self.conversations_dir or os.getcwd(),
                    f"conversation_{self.id}.json",
                )

            # Create directory if it doesn't exist
            save_dir = os.path.dirname(self.save_filepath)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # Save with proper formatting
            with open(self.save_filepath, "w", encoding="utf-8") as f:
                json.dump(
                    self.conversation_history,
                    f,
                    indent=4,
                    default=str,
                )

            logger.info(f"Conversation saved to {self.save_filepath}")

        except Exception as e:
            logger.error(
                f"Failed to save conversation: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
            raise  # Re-raise to ensure the error is visible to the caller

    def save_as_yaml(self, force: bool = True):
        """Save the conversation history and metadata to a YAML file.

        Args:
            force (bool, optional): If True, saves regardless of autosave setting. Defaults to True.
        """
        try:
            # Check if saving is allowed
            if not self.autosave and not force:
                logger.warning(
                    "Autosave is disabled. To save anyway, call save_as_yaml(force=True) "
                    "or enable autosave by setting autosave=True when creating the Conversation."
                )
                return

            # Create directory if it doesn't exist
            save_dir = os.path.dirname(self.save_filepath)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # Save with proper formatting
            with open(self.save_filepath, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.conversation_history,
                    f,
                    indent=4,
                    default_flow_style=False,
                    sort_keys=False,
                )
                logger.info(
                    f"Conversation saved to {self.save_filepath}"
                )

        except Exception as e:
            logger.error(
                f"Failed to save conversation to {self.save_filepath}: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
            raise  # Re-raise the exception to handle it in the calling method

    def export(self, force: bool = True):
        """Export the conversation to a file based on the export method.

        Args:
            force (bool, optional): If True, saves regardless of autosave setting. Defaults to True.
        """
        try:
            # Validate export method
            if self.export_method not in ["json", "yaml"]:
                raise ValueError(
                    f"Invalid export_method: {self.export_method}. Must be 'json' or 'yaml'"
                )

            # Create directory if it doesn't exist
            save_dir = os.path.dirname(self.save_filepath)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            # Ensure filepath extension matches export method
            file_ext = os.path.splitext(self.save_filepath)[1].lower()
            expected_ext = (
                ".json" if self.export_method == "json" else ".yaml"
            )
            if file_ext != expected_ext:
                base_name = os.path.splitext(self.save_filepath)[0]
                self.save_filepath = f"{base_name}{expected_ext}"
                logger.warning(
                    f"Updated save filepath to match export method: {self.save_filepath}"
                )

            if self.export_method == "json":
                self.save_as_json(force=force)
            elif self.export_method == "yaml":
                self.save_as_yaml(force=force)

        except Exception as e:
            logger.error(
                f"Failed to export conversation to {self.save_filepath}: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
            raise  # Re-raise to ensure the error is visible

    def load_from_json(self, filename: str):
        """Load the conversation history and metadata from a JSON file.

        Args:
            filename (str): Filename to load from.
        """
        if filename is not None and os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Load metadata
                metadata = data.get("metadata", {})
                # Update all metadata attributes
                for key, value in metadata.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

                # Load conversation history
                self.conversation_history = data.get(
                    "conversation_history", []
                )

                logger.info(
                    f"Successfully loaded conversation from {filename}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load conversation: {str(e)}\nTraceback: {traceback.format_exc()}"
                )
                raise

    def load_from_yaml(self, filename: str):
        """Load the conversation history and metadata from a YAML file.

        Args:
            filename (str): Filename to load from.
        """
        if filename is not None and os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                # Load metadata
                metadata = data.get("metadata", {})
                # Update all metadata attributes
                for key, value in metadata.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

                # Load conversation history
                self.conversation_history = data.get(
                    "conversation_history", []
                )

                logger.info(
                    f"Successfully loaded conversation from {filename}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load conversation: {str(e)}\nTraceback: {traceback.format_exc()}"
                )
                raise

    def load(self, filename: str):
        """Load the conversation history and metadata from a file.
        Automatically detects the file format based on extension.

        Args:
            filename (str): Filename to load from.
        """
        if filename is None or not os.path.exists(filename):
            logger.warning(f"File not found: {filename}")
            return

        file_ext = os.path.splitext(filename)[1].lower()
        try:
            if file_ext == ".json":
                self.load_from_json(filename)
            elif file_ext == ".yaml" or file_ext == ".yml":
                self.load_from_yaml(filename)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}. Must be .json, .yaml, or .yml"
                )
        except Exception as e:
            logger.error(
                f"Failed to load conversation from {filename}: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
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
        Truncate conversation history based on the total token count using tokenizer.

        This version is more generic, not dependent on a specific LLM model, and can work with any model that provides a counter.
        Uses count_tokens function to calculate and truncate by message, ensuring the result is still valid content.

        Returns:
            None
        """

        total_tokens = 0
        truncated_history = []

        for message in self.conversation_history:
            role = message.get("role")
            content = message.get("content")

            # Convert content to string if it's not already a string
            if not isinstance(content, str):
                content = str(content)

            # Calculate token count for this message
            token_count = count_tokens(
                content, self.tokenizer_model_name
            )

            # Check if adding this message would exceed the limit
            if total_tokens + token_count <= self.context_length:
                # If not exceeding limit, add the full message
                truncated_history.append(message)
                total_tokens += token_count
            else:
                # Calculate remaining tokens we can include
                remaining_tokens = self.context_length - total_tokens

                # If no token space left, break the loop
                if remaining_tokens <= 0:
                    break

                # If we have space left, we need to truncate this message
                # Use binary search to find content length that fits remaining token space
                truncated_content = self._binary_search_truncate(
                    content,
                    remaining_tokens,
                    self.tokenizer_model_name,
                )

                # Create the truncated message
                truncated_message = {
                    "role": role,
                    "content": truncated_content,
                }

                # Add any other fields from the original message
                for key, value in message.items():
                    if key not in ["role", "content"]:
                        truncated_message[key] = value

                truncated_history.append(truncated_message)
                break

        # Update conversation history
        self.conversation_history = truncated_history

    def _binary_search_truncate(
        self, text, target_tokens, model_name
    ):
        """
        Use binary search to find the maximum text substring that fits the target token count.

        Parameters:
            text (str): Original text to truncate
            target_tokens (int): Target token count
            model_name (str): Model name for token counting

        Returns:
            str: Truncated text with token count not exceeding target_tokens
        """

        # If text is empty or target tokens is 0, return empty string
        if not text or target_tokens <= 0:
            return ""

        # If original text token count is already less than or equal to target, return as is
        original_tokens = count_tokens(text, model_name)
        if original_tokens <= target_tokens:
            return text

        # Binary search
        left, right = 0, len(text)
        best_text = ""

        while left <= right:
            mid = (left + right) // 2
            truncated = text[:mid]
            tokens = count_tokens(truncated, model_name)

            if tokens <= target_tokens:
                # If current truncated text token count is less than or equal to target, try longer text
                best_text = truncated
                left = mid + 1
            else:
                # Otherwise try shorter text
                right = mid - 1

        # Try to truncate at sentence boundaries if possible
        sentence_delimiters = [".", "!", "?", "\n"]
        for delimiter in sentence_delimiters:
            last_pos = best_text.rfind(delimiter)
            if (
                last_pos > len(best_text) * 0.75
            ):  # Only truncate at sentence boundary if we don't lose too much content
                truncated_at_sentence = best_text[: last_pos + 1]
                if (
                    count_tokens(truncated_at_sentence, model_name)
                    <= target_tokens
                ):
                    return truncated_at_sentence

        return best_text

    def clear(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def to_json(self):
        """Convert the conversation history to a JSON string.

        Returns:
            str: The conversation history as a JSON string.
        """
        return json.dumps(self.conversation_history)

    def to_list(self):
        """Convert the conversation history to a list.

        Returns:
            list: The conversation history as a list of dictionaries.
        """
        return self.conversation_history

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
        if self.conversation_history:
            return f"{self.conversation_history[-1]['role']}: {self.conversation_history[-1]['content']}"
        return ""

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
        if self.conversation_history:
            return f"{self.conversation_history[-1]['role']}: {self.conversation_history[-1]['content']}"
        return ""

    def get_final_message_content(self):
        """Return the content of the final message from the conversation history.

        Returns:
            str: The content of the final message.
        """
        if self.conversation_history:
            output = self.conversation_history[-1]["content"]
            return output
        return ""

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
            conversation = cls(name=name)
            conversation.load(load_filepath)
            return conversation

        conv_dir = conversations_dir or get_conversation_dir()

        # Try loading by name with different extensions
        for ext in [".json", ".yaml", ".yml"]:
            filepath = os.path.join(conv_dir, f"{name}{ext}")
            if os.path.exists(filepath):
                conversation = cls(
                    name=name, conversations_dir=conv_dir
                )
                conversation.load(filepath)
                return conversation

        # If not found by name with extensions, try loading by ID
        filepath = os.path.join(conv_dir, name)
        if os.path.exists(filepath):
            conversation = cls(name=name, conversations_dir=conv_dir)
            conversation.load(filepath)
            return conversation

        logger.warning(
            f"No conversation found with name or ID: {name}"
        )
        return cls(name=name, conversations_dir=conv_dir)

    def return_dict_final(self):
        """Return the final message as a dictionary."""
        return (
            self.conversation_history[-1]["content"],
            self.conversation_history[-1]["content"],
        )

    def return_list_final(self):
        """Return the final message as a list."""
        return [
            self.conversation_history[-1]["content"],
        ]

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

    def clear_memory(self):
        """Clear the memory of the conversation."""
        self.conversation_history = []

    def _dynamic_auto_chunking_worker(self):
        """
        Dynamically chunk the conversation history to fit within the context length.

        Returns:
            str: The chunked conversation history as a string that fits within context_length tokens.
        """
        all_tokens = self._return_history_as_string_worker()

        total_tokens = count_tokens(
            all_tokens, self.tokenizer_model_name
        )

        if total_tokens <= self.context_length:
            return all_tokens

        # We need to remove characters from the beginning until we're under the limit
        # Start by removing a percentage of characters and adjust iteratively
        target_tokens = self.context_length
        current_string = all_tokens

        # Binary search approach to find the right cutoff point
        left, right = 0, len(all_tokens)

        while left < right:
            mid = (left + right) // 2
            test_string = all_tokens[mid:]

            if not test_string:
                break

            test_tokens = count_tokens(
                test_string, self.tokenizer_model_name
            )

            if test_tokens <= target_tokens:
                # We can remove more from the beginning
                right = mid
                current_string = test_string
            else:
                # We need to keep more from the beginning
                left = mid + 1

        return current_string

    def dynamic_auto_chunking(self):
        """
        Dynamically chunk the conversation history to fit within the context length.

        Returns:
            str: The chunked conversation history as a string that fits within context_length tokens.
        """
        try:
            return self._dynamic_auto_chunking_worker()
        except Exception as e:
            logger.error(f"Dynamic auto chunking failed: {e}")
            return self._return_history_as_string_worker()


# Example usage
# conversation = Conversation()
# conversation = Conversation(token_count=True, context_length=14)
# conversation.add("user", "Hello, how are you?")
# conversation.add("assistant", "I am doing well, thanks.")
# conversation.add("user", "What is the weather in Tokyo?")
# print(conversation.dynamic_auto_chunking())
# # conversation.add(
# #     "assistant", {"name": "tool_1", "output": "Hello, how are you?"}
# )
# print(conversation.return_json())

# # print(conversation.get_last_message_as_string())
# print(conversation.return_json())
# # conversation.add("assistant", "I am doing well, thanks.")
# # # print(conversation.to_json())
# print(type(conversation.to_dict()))
# print(conversation.to_yaml())
