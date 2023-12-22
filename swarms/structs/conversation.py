import json
import datetime

from termcolor import colored

from swarms.structs.base import BaseStructure


class Conversation(BaseStructure):
    """
    Conversation class
    
    
    Attributes:
        time_enabled (bool): whether to enable time
        conversation_history (list): list of messages in the conversation
        
    
    Examples:
    >>> conv = Conversation()
    >>> conv.add("user", "Hello, world!")
    >>> conv.add("assistant", "Hello, user!")
    >>> conv.display_conversation()
    user: Hello, world!
    
    
    """
    def __init__(self, time_enabled: bool = False, *args, **kwargs):
        super().__init__()
        self.time_enabled = time_enabled
        self.conversation_history = []

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

    def export_conversation(self, filename: str):
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
        with open(filename, "r") as f:
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

    def save_as_json(self, filename: str):
        """Save the conversation history as a JSON file

        Args:
            filename (str): Save the conversation history as a JSON file
        """
        # Save the conversation history as a JSON file
        with open(filename, "w") as f:
            json.dump(self.conversation_history, f)

    def load_from_json(self, filename: str):
        """Load the conversation history from a JSON file

        Args:
            filename (str): filename to load from
        """
        # Load the conversation history from a JSON file
        with open(filename, "r") as f:
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
