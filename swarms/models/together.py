import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

from swarms.models.base_llm import BaseLLM

# Load environment variables
load_dotenv()


def together_api_key_env():
    """Get the API key from the environment."""
    return os.getenv("TOGETHER_API_KEY")


class TogetherLLM(BaseLLM):
    """
    GPT-4 Vision API

    This class is a wrapper for the OpenAI API. It is used to run the GPT-4 Vision model.

    Parameters
    ----------
    together_api_key : str
        The OpenAI API key. Defaults to the together_api_key environment variable.
    max_tokens : int
        The maximum number of tokens to generate. Defaults to 300.


    Methods
    -------
    encode_image(img: str)
        Encode image to base64.
    run(task: str, img: str)
        Run the model.
    __call__(task: str, img: str)
        Run the model.

    Examples:
    ---------
    >>> from swarms.models import GPT4VisionAPI
    >>> llm = GPT4VisionAPI()
    >>> task = "What is the color of the object?"
    >>> img = "https://i.imgur.com/2M2ZGwC.jpeg"
    >>> llm.run(task, img)


    """

    def __init__(
        self,
        together_api_key: str = together_api_key_env,
        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        logging_enabled: bool = False,
        max_workers: int = 10,
        max_tokens: str = 300,
        api_endpoint: str = "https://api.together.xyz",
        beautify: bool = False,
        streaming_enabled: Optional[bool] = False,
        meta_prompt: Optional[bool] = False,
        system_prompt: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super(TogetherLLM).__init__(*args, **kwargs)
        self.together_api_key = together_api_key
        self.logging_enabled = logging_enabled
        self.model_name = model_name
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.api_endpoint = api_endpoint
        self.beautify = beautify
        self.streaming_enabled = streaming_enabled
        self.meta_prompt = meta_prompt
        self.system_prompt = system_prompt

        if self.logging_enabled:
            logging.basicConfig(level=logging.DEBUG)
        else:
            # Disable debug logs for requests and urllib3
            logging.getLogger("requests").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)

        if self.meta_prompt:
            self.system_prompt = self.meta_prompt_init()

    # Function to handle vision tasks
    def run(self, task: str = None, *args, **kwargs):
        """Run the model."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.together_api_key}",
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": [self.system_prompt],
                    },
                    {
                        "role": "user",
                        "content": task,
                    },
                ],
                "max_tokens": self.max_tokens,
                **kwargs,
            }
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload,
                *args,
                **kwargs,
            )

            out = response.json()
            content = (
                out["choices"][0].get("message", {}).get("content", None)
            )
            if self.streaming_enabled:
                content = self.stream_response(content)

            return content

        except Exception as error:
            print(
                f"Error with the request: {error}, make sure you"
                " double check input types and positions"
            )
            return None
