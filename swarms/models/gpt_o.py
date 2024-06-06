import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

from swarms.models.base_multimodal_model import BaseMultiModalModel

# Load the OpenAI API key from the .env file
load_dotenv()

# Initialize the OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class GPT4o(BaseMultiModalModel):
    """
    GPT4o is a class that represents a multi-modal conversational model based on GPT-4.
    It extends the BaseMultiModalModel class.

    Args:
        system_prompt (str): The system prompt to be used in the conversation.
        temperature (float): The temperature parameter for generating diverse responses.
        max_tokens (int): The maximum number of tokens in the generated response.
        openai_api_key (str): The API key for accessing the OpenAI GPT-4 API.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Attributes:
        system_prompt (str): The system prompt to be used in the conversation.
        temperature (float): The temperature parameter for generating diverse responses.
        max_tokens (int): The maximum number of tokens in the generated response.
        client (OpenAI): The OpenAI client for making API requests.

    Methods:
        run(task, local_img=None, img=None, *args, **kwargs):
            Runs the GPT-4o model to generate a response based on the given task and image.

    """

    def __init__(
        self,
        system_prompt: str = None,
        temperature: float = 0.1,
        max_tokens: int = 300,
        openai_api_key: str = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(api_key=openai_api_key, *args, **kwargs)

    def run(
        self,
        task: str,
        local_img: str = None,
        img: str = None,
        *args,
        **kwargs,
    ):
        """
        Runs the GPT-4o model to generate a response based on the given task and image.

        Args:
            task (str): The task or user prompt for the conversation.
            local_img (str): The local path to the image file.
            img (str): The URL of the image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated response from the GPT-4o model.

        """
        img = encode_image(local_img)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content
