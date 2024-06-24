import os
from swarms.models.base_llm import BaseLLM


def check_multion_api_key():
    """
    Checks if the MultiOn API key is available in the environment variables.

    Returns:
        str: The MultiOn API key.
    """
    api_key = os.getenv("MULTION_API_KEY")
    return api_key


class MultiOnAgent(BaseLLM):
    """
    Represents an agent that interacts with the MultiOn API to run tasks on a remote session.

    Args:
        api_key (str): The API key for accessing the MultiOn API.
        url (str): The URL of the remote session.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        client (MultiOn): The MultiOn client instance.
        url (str): The URL of the remote session.
        session_id (str): The ID of the current session.

    Methods:
        run: Runs a task on the remote session.
    """

    def __init__(
        self,
        name: str = None,
        system_prompt: str = None,
        api_key: str = check_multion_api_key,
        url: str = "https://huggingface.co/papers",
        max_steps: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name

        try:
            from multion.client import MultiOn
        except ImportError:
            raise ImportError(
                "The MultiOn package is not installed. Please install it using 'pip install multion'."
            )

        self.client = MultiOn(api_key=api_key)
        self.url = url
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, task: str, *args, **kwargs):
        """
        Runs a task on the remote session.

        Args:
            task (str): The task to be executed on the remote session.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        response = self.client.browse(
            cmd=task,
            url=self.url,
            local=True,
            max_steps=self.max_steps,
        )

        # response = response.json()

        # print(response.message)
        return str(response.message)
