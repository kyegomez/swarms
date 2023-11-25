import base64
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class GPT4VisionAPI:
    """
    GPT-4 Vision API

    This class is a wrapper for the OpenAI API. It is used to run the GPT-4 Vision model.

    Parameters
    ----------
    openai_api_key : str
        The OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.
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

    def __init__(self, openai_api_key: str = openai_api_key, max_tokens: str = 300):
        super().__init__()
        self.openai_api_key = openai_api_key
        self.max_tokens = max_tokens

    def encode_image(self, img: str):
        """Encode image to base64."""
        with open(img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Function to handle vision tasks
    def run(self, task: str, img: str):
        """Run the model."""
        try:
            base64_image = self.encode_image(img)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}",
            }
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": task},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:image/jpeg;base64,{base64_image}"
                                    )
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": self.max_tokens,
            }
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            out = response.json()
            content = out["choices"][0]["message"]["content"]
            print(content)
        except Exception as error:
            print(f"Error with the request: {error}")
            raise error
        # Function to handle vision tasks

    def __call__(self, task: str, img: str):
        """Run the model."""
        try:
            base64_image = self.encode_image(img)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}",
            }
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": task},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:image/jpeg;base64,{base64_image}"
                                    )
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": self.max_tokens,
            }
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            out = response.json()
            content = out["choices"][0]["message"]["content"]
            print(content)
        except Exception as error:
            print(f"Error with the request: {error}")
            raise error
        # Function to handle vision tasks