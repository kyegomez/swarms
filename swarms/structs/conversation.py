import datetime
import json
from typing import Optional

from termcolor import colored

from swarms.structs.base_structure import BaseStructure
from typing import Any


class Conversation(BaseStructure):
    """
    A class structure to represent a conversation in a chatbot. This class is used to store the conversation history.
    And, it can be used to save the conversation history to a file, load the conversation history from a file, and
    display the conversation history. We can also use this class to add the conversation history to a database, query
    the conversation history from a database, delete the conversation history from a database, update the conversation
    history from a database, and get the conversation history from a database.


    Args:
        time_enabled (bool): Whether to enable timestamps for the conversation history. Default is False.
        database (AbstractDatabase): The database to use for storing the conversation history. Default is None.
        autosave (bool): Whether to autosave the conversation history to a file. Default is None.
        save_filepath (str): The filepath to save the conversation history to. Default is None.


    Methods:
        add(role: str, content: str): Add a message to the conversation history.
        delete(index: str): Delete a message from the conversation history.
        update(index: str, role, content): Update a message in the conversation history.
        query(index: str): Query a message in the conversation history.
        search(keyword: str): Search for a message in the conversation history.
        display_conversation(detailed: bool = False): Display the conversation history.
        export_conversation(filename: str): Export the conversation history to a file.
        import_conversation(filename: str): Import a conversation history from a file.
        count_messages_by_role(): Count the number of messages by role.
        return_history_as_string(): Return the conversation history as a string.
        save_as_json(filename: str): Save the conversation history as a JSON file.
        load_from_json(filename: str): Load the conversation history from a JSON file.
        search_keyword_in_conversation(keyword: str): Search for a keyword in the conversation history.
        pretty_print_conversation(messages): Pretty print the conversation history.
        add_to_database(): Add the conversation history to the database.
        query_from_database(query): Query the conversation history from the database.
        delete_from_database(): Delete the conversation history from the database.
        update_from_database(): Update the conversation history from the database.
        get_from_database(): Get the conversation history from the database.
        execute_query_from_database(query): Execute a query on the database.
        fetch_all_from_database(): Fetch all from the database.
        fetch_one_from_database(): Fetch one from the database.

    Examples:
        >>> from swarms import Conversation
        >>> conversation = Conversation()
        >>> conversation.add("user", "Hello, how are you?")
        >>> conversation.add("assistant", "I am doing well, thanks.")
        >>> conversation.display_conversation()
        user: Hello, how are you?
        assistant: I am doing well, thanks.

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
        *args,
        **kwargs,
    ):
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

        # If system prompt is not None, add it to the conversation history
        if self.system_prompt is not None:
            self.add("System: ", self.system_prompt)

        if self.rules is not None:
            self.add(user, rules)

        if custom_rules_prompt is not None:
            self.add(user, custom_rules_prompt)

        # If tokenizer then truncate
        if tokenizer is not None:
            self.truncate_memory_with_tokenizer()

    def add(self, role: str, content: str, *args, **kwargs):
        """Add a message to the conversation history

        Args:
            role (str): The role of the speaker
            content (str): The content of the message

        """
        if self.time_enabled:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            message = {
                "role": role,
                "content": content,
                "timestamp": timestamp,
            }
        else:
            message = {
                "role": role,
                "content": content,
            }

        self.conversation_history.append(message)

        if self.autosave:
            self.save_as_json(self.save_filepath)

    def delete(self, index: str):
        """Delete a message from the conversation history

        Args:
            index (str): index of the message to delete
        """
        self.conversation_history.pop(index)

    def update(self, index: str, role, content):
        """Update a message in the conversation history

        Args:
            index (str): index of the message to update
            role (_type_): role of the speaker
            content (_type_): content of the message
        """
        self.conversation_history[index] = {
            "role": role,
            "content": content,
        }

    def query(self, index: str):
        """Query a message in the conversation history

        Args:
            index (str): index of the message to query

        Returns:
            str: the message
        """
        return self.conversation_history[index]

    def search(self, keyword: str):
        """Search for a message in the conversation history

        Args:
            keyword (str): Keyword to search for

        Returns:
            str: description
        """
        return [
            msg
            for msg in self.conversation_history
            if keyword in msg["content"]
        ]

    def display_conversation(self, detailed: bool = False):
        """Display the conversation history

        Args:
            detailed (bool, optional): detailed. Defaults to False.
        """
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )

    def export_conversation(self, filename: str, *args, **kwargs):
        """Export the conversation history to a file

        Args:
            filename (str): filename to export to
        """
        with open(filename, "w") as f:
            for message in self.conversation_history:
                f.write(f"{message['role']}: {message['content']}\n")

    def import_conversation(self, filename: str):
        """Import a conversation history from a file

        Args:
            filename (str): filename to import from
        """
        with open(filename) as f:
            for line in f:
                role, content = line.split(": ", 1)
                self.add(role, content.strip())

    def count_messages_by_role(self):
        """Count the number of messages by role"""
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
        """Return the conversation history as a string

        Returns:
            str: the conversation history
        """
        return "\n".join(
            [
                f"{message['role']}: {message['content']}\n\n"
                for message in self.conversation_history
            ]
        )

    def save_as_json(self, filename: str = None):
        """Save the conversation history as a JSON file

        Args:
            filename (str): Save the conversation history as a JSON file
        """
        # Create the directory if it does not exist
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        if filename is not None:
            with open(filename, "w") as f:
                json.dump(self.conversation_history, f)

    def load_from_json(self, filename: str):
        """Load the conversation history from a JSON file

        Args:
            filename (str): filename to load from
        """
        # Load the conversation history from a JSON file
        if filename is not None:
            with open(filename) as f:
                self.conversation_history = json.load(f)

    def search_keyword_in_conversation(self, keyword: str):
        """Search for a keyword in the conversation history

        Args:
            keyword (str): keyword to search for

        Returns:
            str: description
        """
        return [
            msg
            for msg in self.conversation_history
            if keyword in msg["content"]
        ]

    def pretty_print_conversation(self, messages):
        """Pretty print the conversation history

        Args:
            messages (str): messages to print
        """
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "tool": "magenta",
        }

        for message in messages:
            if message["role"] == "system":
                print(
                    colored(
                        f"system: {message['content']}\n",
                        role_to_color[message["role"]],
                    )
                )
            elif message["role"] == "user":
                print(
                    colored(
                        f"user: {message['content']}\n",
                        role_to_color[message["role"]],
                    )
                )
            elif message["role"] == "assistant" and message.get(
                "function_call"
            ):
                print(
                    colored(
                        f"assistant: {message['function_call']}\n",
                        role_to_color[message["role"]],
                    )
                )
            elif message["role"] == "assistant" and not message.get(
                "function_call"
            ):
                print(
                    colored(
                        f"assistant: {message['content']}\n",
                        role_to_color[message["role"]],
                    )
                )
            elif message["role"] == "tool":
                print(
                    colored(
                        (
                            f"function ({message['name']}):"
                            f" {message['content']}\n"
                        ),
                        role_to_color[message["role"]],
                    )
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
        self.conversation_history = []
