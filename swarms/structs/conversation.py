import datetime
import json
from typing import Any, Optional, Union

import yaml
from swarms.structs.base_structure import BaseStructure
from typing import TYPE_CHECKING
from swarms.utils.any_to_str import any_to_str
from swarms.utils.formatter import formatter
from swarms.utils.litellm_tokenizer import count_tokens
import threading

if TYPE_CHECKING:
    from swarms.structs.agent import (
        Agent,
    )  # Only imported during type checking


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
        *args,
        **kwargs,
    ):
        """
        Initializes the Conversation object with the provided parameters.

        Args:
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
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.time_enabled = time_enabled
        self.autosave = autosave
        self.save_filepath = save_filepath
        self.conversation_history = []
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.rules = rules
        self.custom_rules_prompt = custom_rules_prompt
        self.user = user
        self.auto_save = auto_save
        self.save_as_yaml = save_as_yaml
        self.save_as_json_bool = save_as_json_bool
        self.token_count = token_count

        # If system prompt is not None, add it to the conversation history
        if self.system_prompt is not None:
            self.add("System: ", self.system_prompt)

        if self.rules is not None:
            self.add("User", rules)

        if custom_rules_prompt is not None:
            self.add(user or "User", custom_rules_prompt)

        # If tokenizer then truncate
        if tokenizer is not None:
            self.truncate_memory_with_tokenizer()

    def add(
        self,
        role: str,
        content: Union[str, dict, list],
        *args,
        **kwargs,
    ):
        """Add a message to the conversation history.

        Args:
            role (str): The role of the speaker (e.g., 'User', 'System').
            content (Union[str, dict, list]): The content of the message to be added.
        """
        now = datetime.datetime.now()
        now.strftime("%Y-%m-%d %H:%M:%S")

        # Base message with role
        message = {
            "role": role,
        }

        # Handle different content types
        if isinstance(content, dict) or isinstance(content, list):
            message["content"] = content
        elif self.time_enabled:
            message["content"] = (
                f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')} \n {content}"
            )
        else:
            message["content"] = content

        # Add the message to history immediately without waiting for token count
        self.conversation_history.append(message)

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
        elif self.autosave:
            # If token counting is disabled but autosave is enabled, save immediately
            self.save_as_json(self.save_filepath)

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
            formatter.print_panel(
                f"{message['role']}: {message['content']}\n\n"
            )

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
        return "\n".join(
            [
                f"{message['role']}: {message['content']}\n\n"
                for message in self.conversation_history
            ]
        )

    def get_str(self):
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
        if filename is not None:
            with open(filename, "w") as f:
                json.dump(self.conversation_history, f)

    def load_from_json(self, filename: str):
        """Load the conversation history from a JSON file.

        Args:
            filename (str): Filename to load from.
        """
        if filename is not None:
            with open(filename) as f:
                self.conversation_history = json.load(f)

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

    def pretty_print_conversation(self, messages):
        """Pretty print the conversation history.

        Args:
            messages (list): List of messages to print.
        """
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "tool": "magenta",
        }

        for message in messages:
            if message["role"] == "system":
                formatter.print_panel(
                    f"system: {message['content']}\n",
                    role_to_color[message["role"]],
                )
            elif message["role"] == "user":
                formatter.print_panel(
                    f"user: {message['content']}\n",
                    role_to_color[message["role"]],
                )
            elif message["role"] == "assistant" and message.get(
                "function_call"
            ):
                formatter.print_panel(
                    f"assistant: {message['function_call']}\n",
                    role_to_color[message["role"]],
                )
            elif message["role"] == "assistant" and not message.get(
                "function_call"
            ):
                formatter.print_panel(
                    f"assistant: {message['content']}\n",
                    role_to_color[message["role"]],
                )
            elif message["role"] == "tool":
                formatter.print_panel(
                    (
                        f"function ({message['name']}):"
                        f" {message['content']}\n"
                    ),
                    role_to_color[message["role"]],
                )

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
        return json.dumps(self.conversation_history)

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
        return f"{self.conversation_history[-1]['role']}: {self.conversation_history[-1]['content']}"

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
