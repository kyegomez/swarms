import json
import time
import uuid

import yaml
from dotenv import load_dotenv
from loguru import logger

from swarms.structs.agent import Agent
from swarms.prompts.prompt_generator_optimizer import (
    prompt_generator_sys_prompt,
)


load_dotenv()


class PromptGeneratorAgent:
    """
    A class representing a prompt generator agent.

    Attributes:
    ----------
    agent : Agent
        The underlying agent instance.
    """

    def __init__(self, agent: Agent):
        """
        Initializes the PromptGeneratorAgent instance.

        Args:
        ----
        agent : Agent
            The agent instance to be used for prompt generation.
        """
        self.agent = agent

    def run(self, task: str, format: str = "json") -> str:
        """
        Runs the prompt generator agent with the given task description and saves the generated prompt with the given metadata in the specified format.

        Args:
        ----
        task : str
            The task description to be used for prompt generation.
        metadata : Dict[str, Any]
            The metadata to be saved along with the prompt.
        format : str, optional
            The format in which the prompt should be saved (default is "json").

        Returns:
        -------
        str
            The generated prompt.
        """
        prompt = self.agent.run(task)
        self.save_prompt(prompt, format)
        return prompt

    def save_prompt(
        self,
        prompt: str,
        format: str = "yaml",
    ):
        """
        Saves the generated prompt with the given metadata in the specified format using the prompt generator sys prompt model dump.

        Args:
        ----
        prompt : str
            The generated prompt to be saved.
        metadata : Dict[str, Any]
            The metadata to be saved along with the prompt.
        format : str, optional
            The format in which the prompt should be saved (default is "json").
        """
        data = {
            "prompt_history": prompt_generator_sys_prompt.model_dump(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
        }
        if format == "json":
            with open(f"prompt_{uuid.uuid4()}.json", "w") as f:
                json.dump(data, f, indent=4)
        elif format == "yaml":
            with open(f"prompt_{uuid.uuid4()}.yaml", "w") as f:
                yaml.dump(data, f)
        else:
            logger.error(
                "Invalid format. Only 'json' and 'yaml' are supported."
            )
