import os
import multion

from swarms.models.base_llm import AbstractLLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Muliton key
MULTION_API_KEY = os.getenv("MULTION_API_KEY")


class MultiOnAgent(AbstractLLM):
    """
    Represents a multi-on agent that performs browsing tasks.

    Args:
        max_steps (int): The maximum number of steps to perform during browsing.
        starting_url (str): The starting URL for browsing.

    Attributes:
        max_steps (int): The maximum number of steps to perform during browsing.
        starting_url (str): The starting URL for browsing.
    """

    def __init__(
        self,
        multion_api_key: str = MULTION_API_KEY,
        max_steps: int = 4,
        starting_url: str = "https://www.google.com",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multion_api_key = multion_api_key
        self.max_steps = max_steps
        self.starting_url = starting_url

    def run(self, task: str, *args, **kwargs):
        """
        Runs a browsing task.

        Args:
            task (str): The task to perform during browsing.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The response from the browsing task.
        """
        multion.login(
            use_api=True,
            multion_api_key=str(self.multion_api_key),
            *args,
            **kwargs,
        )

        response = multion.browse(
            {
                "cmd": task,
                "url": self.starting_url,
                "maxSteps": self.max_steps,
            },
            *args,
            **kwargs,
        )

        return response.result, response.status, response.lastUrl
