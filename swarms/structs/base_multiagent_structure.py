import json
from typing import List, Optional, Sequence

import yaml

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.logger import logger


class BaseMultiAgentStructure:
    """
    Base class for a multi-agent structure.

    Args:
        agents (List[Agent], optional): List of agents in the structure. Defaults to None.
        callbacks (Optional[Sequence[callable]], optional): List of callbacks for the structure. Defaults to None.
        autosave (bool, optional): Flag indicating whether to enable autosave. Defaults to False.
        logging (bool, optional): Flag indicating whether to enable logging. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        agents (List[Agent]): List of agents in the structure.
        callbacks (Optional[Sequence[callable]]): List of callbacks for the structure.
        autosave (bool): Flag indicating whether autosave is enabled.
        logging (bool): Flag indicating whether logging is enabled.
        conversation (Conversation): Conversation object for the structure.

    Methods:
        metadata(): Get the metadata of the multi-agent structure.
        save_to_json(filename: str): Save the current state of the multi-agent structure to a JSON file.
        load_from_json(filename: str): Load the state of the multi-agent structure from a JSON file.
    """

    def __init__(
        self,
        agents: List[Agent] = None,
        callbacks: Optional[Sequence[callable]] = None,
        autosave: bool = False,
        logging: bool = False,
        return_metadata: bool = False,
        metadata_filename: str = "multiagent_structure_metadata.json",
        *args,
        **kwargs,
    ):
        self.agents = agents
        self.callbacks = callbacks
        self.autosave = autosave
        self.logging = logging
        self.return_metadata = return_metadata
        self.metadata_filename = metadata_filename
        self.conversation = Conversation(
            time_enabled=True, *args, **kwargs
        )
        if self.logging:
            self.logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

        # Handle the case where the agents are not provided
        # Handle agents
        for agent in self.agents:
            if not isinstance(agent, Agent):
                raise TypeError("Agents must be of type Agent.")

        if self.agents is None:
            self.agents = []

        # Handle the case where the callbacks are not provided
        if self.callbacks is None:
            self.callbacks = []

        # Handle the case where the autosave is not provided
        if self.autosave is None:
            self.autosave = False

        # Handle the case where the logging is not provided
        if self.logging is None:
            self.logging = False

        # Handle callbacks
        if callbacks is not None:
            for callback in self.callbacks:
                if not callable(callback):
                    raise TypeError("Callback must be callable.")

        # Handle autosave
        if autosave:
            self.save_to_json(metadata_filename)

    def metadata(self):
        """
        Get the metadata of the multi-agent structure.

        Returns:
            dict: The metadata of the multi-agent structure.
        """
        return {
            "agents": self.agents,
            "callbacks": self.callbacks,
            "autosave": self.autosave,
            "logging": self.logging,
            "conversation": self.conversation,
        }

    def save_to_json(self, filename: str):
        """
        Save the current state of the multi-agent structure to a JSON file.

        Args:
            filename (str): The name of the file to save the multi-agent structure to.

        Returns:
            None
        """
        try:
            with open(filename, "w") as f:
                json.dump(self.__dict__, f)
        except Exception as e:
            logger.error(e)

    def load_from_json(self, filename: str):
        """
        Load the state of the multi-agent structure from a JSON file.

        Args:
            filename (str): The name of the file to load the multi-agent structure from.

        Returns:
            None
        """
        try:
            with open(filename) as f:
                self.__dict__ = json.load(f)
        except Exception as e:
            logger.error(e)

    def save_to_yaml(self, filename: str):
        """
        Save the current state of the multi-agent structure to a YAML file.

        Args:
            filename (str): The name of the file to save the multi-agent structure to.

        Returns:
            None
        """
        try:
            with open(filename, "w") as f:
                yaml.dump(self.__dict__, f)
        except Exception as e:
            logger.error(e)

    def load_from_yaml(self, filename: str):
        """
        Load the state of the multi-agent structure from a YAML file.

        Args:
            filename (str): The name of the file to load the multi-agent structure from.

        Returns:
            None
        """
        try:
            with open(filename) as f:
                self.__dict__ = yaml.load(f)
        except Exception as e:
            logger.error(e)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, index):
        return self.agents[index]

    def __setitem__(self, index, value):
        self.agents[index] = value

    def __delitem__(self, index):
        del self.agents[index]

    def __iter__(self):
        return iter(self.agents)

    def __reversed__(self):
        return reversed(self.agents)

    def __contains__(self, value):
        return value in self.agents
