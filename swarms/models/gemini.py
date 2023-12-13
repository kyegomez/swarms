import os
import subprocess as sp
from pathlib import Path

from dotenv import load_dotenv

from swarms.models.base_multimodal_model import BaseMultiModalModel

try:
    import google.generativeai as genai
except ImportError as error:
    print(f"Error importing google.generativeai: {error}")
    print("Please install the google.generativeai package")
    print("pip install google-generativeai")
    sp.run(["pip", "install", "--upgrade", "google-generativeai"])


load_dotenv()


# Helpers
def get_gemini_api_key_env():
    """Get the Gemini API key from the environment

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    key = os.getenv("GEMINI_API_KEY")
    if key is None:
        raise ValueError("Please provide a Gemini API key")
    return key


# Main class
class Gemini(BaseMultiModalModel):
    """Gemini model

    Args:
        BaseMultiModalModel (class): Base multimodal model class
        model_name (str, optional): model name. Defaults to "gemini-pro".
        gemini_api_key (str, optional): Gemini API key. Defaults to None.

    Methods:
        run: run the Gemini model
        process_img: process the image


    Examples:
        >>> from swarms.models import Gemini
        >>> gemini = Gemini()
        >>> gemini.run(
                task="A dog",
                img="dog.png",
            )
    """

    def __init__(
        self,
        model_name: str = "gemini-pro",
        gemini_api_key: str = get_gemini_api_key_env,
        *args,
        **kwargs,
    ):
        super().__init__(model_name, *args, **kwargs)
        self.model_name = model_name
        self.gemini_api_key = gemini_api_key

        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name, *args, **kwargs
        )

    def run(
        self,
        task: str = None,
        img: str = None,
        *args,
        **kwargs,
    ) -> str:
        """Run the Gemini model

        Args:
            task (str, optional): textual task. Defaults to None.
            img (str, optional): img. Defaults to None.

        Returns:
            str: output from the model
        """
        try:
            if img:
                process_img = self.process_img(img, *args, **kwargs)
                response = self.model.generate_content(
                    content=[task, process_img], *args, **kwargs
                )
                return response.text
            else:
                response = self.model.generate_content(
                    task, *args, **kwargs
                )
                return response
        except Exception as error:
            print(f"Error running Gemini model: {error}")

    def process_img(
        self,
        img: str = None,
        type: str = "image/png",
        *args,
        **kwargs,
    ):
        """Process the image

        Args:
            img (str, optional): _description_. Defaults to None.
            type (str, optional): _description_. Defaults to "image/png".

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        try:
            if img is None:
                raise ValueError("Please provide an image to process")
            if type is None:
                raise ValueError("Please provide the image type")
            if self.gemini_api_key is None:
                raise ValueError("Please provide a Gemini API key")

            # Load the image
            img = [
                {"mime_type": type, "data": Path(img).read_bytes()}
            ]
        except Exception as error:
            print(f"Error processing image: {error}")

    def chat(
        self,
        task: str = None,
        img: str = None,
        *args,
        **kwargs,
    ) -> str:
        """Chat with the Gemini model

        Args:
            task (str, optional): _description_. Defaults to None.
            img (str, optional): _description_. Defaults to None.

        Returns:
            str: _description_
        """
        chat = self.model.start_chat()
        response = chat.send_message(task, *args, **kwargs)
        response1 = response.text
        print(response1)
        response = chat.send_message(img, *args, **kwargs)
