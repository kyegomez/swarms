import json
import logging
import threading

from swarms.structs.base_structure import BaseStructure


class ShortTermMemory(BaseStructure):
    """Short term memory.

    Args:
    return_str (bool, optional): _description_. Defaults to True.
    autosave (bool, optional): _description_. Defaults to True.
    *args: _description_
    **kwargs: _description_


    Example:
    >>> from swarms.memory.short_term_memory import ShortTermMemory
    >>> stm = ShortTermMemory()
    >>> stm.add(role="agent", message="Hello world!")
    >>> stm.add(role="agent", message="How are you?")
    >>> stm.add(role="agent", message="I am fine.")
    >>> stm.add(role="agent", message="How are you?")
    >>> stm.add(role="agent", message="I am fine.")


    """

    def __init__(
        self,
        return_str: bool = True,
        autosave: bool = True,
        *args,
        **kwargs,
    ):
        self.return_str = return_str
        self.autosave = autosave
        self.short_term_memory = []
        self.medium_term_memory = []
        self.lock = threading.Lock()

    def add(self, role: str = None, message: str = None, *args, **kwargs):
        """Add a message to the short term memory.

        Args:
            role (str, optional): _description_. Defaults to None.
            message (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        try:
            memory = self.short_term_memory.append(
                {"role": role, "message": message}
            )

            return memory
        except Exception as error:
            print(f"Add to short term memory failed: {error}")
            raise error

    def get_short_term(self):
        """Get the short term memory.

        Returns:
            _type_: _description_
        """
        return self.short_term_memory

    def get_medium_term(self):
        """Get the medium term memory.

        Returns:
            _type_: _description_
        """
        return self.medium_term_memory

    def clear_medium_term(self):
        """Clear the medium term memory."""
        self.medium_term_memory = []

    def get_short_term_memory_str(self, *args, **kwargs):
        """Get the short term memory as a string."""
        return str(self.short_term_memory)

    def update_short_term(
        self, index, role: str, message: str, *args, **kwargs
    ):
        """Update the short term memory.

        Args:
            index (_type_): _description_
            role (str): _description_
            message (str): _description_

        """
        self.short_term_memory[index] = {
            "role": role,
            "message": message,
        }

    def clear(self):
        """Clear the short term memory."""
        self.short_term_memory = []

    def search_memory(self, term):
        """Search the memory for a term.

        Args:
            term (_type_): _description_

        Returns:
            _type_: _description_
        """
        results = {"short_term": [], "medium_term": []}
        for i, message in enumerate(self.short_term_memory):
            if term in message["message"]:
                results["short_term"].append((i, message))
        for i, message in enumerate(self.medium_term_memory):
            if term in message["message"]:
                results["medium_term"].append((i, message))
        return results

    def return_shortmemory_as_str(self):
        """Return the memory as a string.

        Returns:
            _type_: _description_
        """
        return str(self.short_term_memory)

    def move_to_medium_term(self, index):
        """Move a message from the short term memory to the medium term memory.

        Args:
            index (_type_): _description_
        """
        message = self.short_term_memory.pop(index)
        self.medium_term_memory.append(message)

    def return_medium_memory_as_str(self):
        """Return the medium term memory as a string.

        Returns:
            _type_: _description_
        """
        return str(self.medium_term_memory)

    def save_to_file(self, filename: str):
        """Save the memory to a file.

        Args:
            filename (str): _description_
        """
        try:
            with self.lock:
                with open(filename, "w") as f:
                    json.dump(
                        {
                            "short_term_memory": (self.short_term_memory),
                            "medium_term_memory": (
                                self.medium_term_memory
                            ),
                        },
                        f,
                    )

                    logging.info(f"Saved memory to {filename}")
        except Exception as error:
            print(f"Error saving memory to {filename}: {error}")

    def load_from_file(self, filename: str, *args, **kwargs):
        """Load the memory from a file.

        Args:
            filename (str): _description_
        """
        try:
            with self.lock:
                with open(filename) as f:
                    data = json.load(f)
                self.short_term_memory = data.get("short_term_memory", [])
                self.medium_term_memory = data.get(
                    "medium_term_memory", []
                )
                logging.info(f"Loaded memory from {filename}")
        except Exception as error:
            print(f"Erorr loading memory from {filename}: {error}")
