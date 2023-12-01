import base64
import os
import requests
import uuid
import shutil
from dotenv import load_dotenv
from typing import List

load_dotenv()


class StableDiffusion:
    """
    A class to interact with the Stable Diffusion API for generating images from text prompts.

    Attributes:
    -----------
    api_key : str
        The API key for accessing the Stable Diffusion API.
    api_host : str
        The host URL for the Stable Diffusion API.
    engine_id : str
        The engine ID for the Stable Diffusion API.
    cfg_scale : int
        Configuration scale for image generation.
    height : int
        The height of the generated image.
    width : int
        The width of the generated image.
    samples : int
        The number of samples to generate.
    steps : int
        The number of steps for the generation process.
    output_dir : str
        Directory where the generated images will be saved.

    Methods:
    --------
    __init__(self, api_key: str, api_host: str, cfg_scale: int, height: int, width: int, samples: int, steps: int):
        Initializes the StableDiffusion instance with provided parameters.

    generate_image(self, task: str) -> List[str]:
        Generates an image based on the provided text prompt and returns the paths of the saved images.
    """

    def __init__(
        self,
        api_key: str,
        api_host: str = "https://api.stability.ai",
        cfg_scale: int = 7,
        height: int = 1024,
        width: int = 1024,
        samples: int = 1,
        steps: int = 30,
    ):
        """
        Initialize the StableDiffusion class with API configurations.

        Parameters:
        -----------
        api_key : str
            The API key for accessing the Stable Diffusion API.
        api_host : str
            The host URL for the Stable Diffusion API.
        cfg_scale : int
            Configuration scale for image generation.
        height : int
            The height of the generated image.
        width : int
            The width of the generated image.
        samples : int
            The number of samples to generate.
        steps : int
            The number of steps for the generation process.
        """
        self.api_key = api_key
        self.api_host = api_host
        self.engine_id = "stable-diffusion-v1-6"
        self.cfg_scale = cfg_scale
        self.height = height
        self.width = width
        self.samples = samples
        self.steps = steps
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.output_dir = "images"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, task: str) -> List[str]:
        """
        Generates an image based on a given text prompt.

        Parameters:
        -----------
        task : str
            The text prompt based on which the image will be generated.

        Returns:
        --------
        List[str]:
            A list of file paths where the generated images are saved.

        Raises:
        -------
        Exception:
            If the API request fails and returns a non-200 response.
        """
        response = requests.post(
            f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image",
            headers=self.headers,
            json={
                "text_prompts": [{"text": task}],
                "cfg_scale": self.cfg_scale,
                "height": self.height,
                "width": self.width,
                "samples": self.samples,
                "steps": self.steps,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Non-200 response: {response.text}")

        data = response.json()
        image_paths = []
        for i, image in enumerate(data["artifacts"]):
            unique_id = uuid.uuid4()  # Generate a unique identifier
            image_path = os.path.join(
                self.output_dir, f"{unique_id}_v1_txt2img_{i}.png"
            )
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image["base64"]))
            image_paths.append(image_path)

        return image_paths

    def generate_and_move_image(self, prompt, iteration, folder_path):
        # Generate the image
        image_paths = self.run(prompt)
        if not image_paths:
            return None

        # Move the image to the specified folder
        src_image_path = image_paths[0]
        dst_image_path = os.path.join(
            folder_path, f"image_{iteration}.jpg"
        )
        shutil.move(src_image_path, dst_image_path)
        return dst_image_path
