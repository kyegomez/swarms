import os
import base64
import requests
from dotenv import load_dotenv
from typing import List

load_dotenv()

class StableDiffusion:
    """
    A class to interact with the Stable Diffusion API for image generation.

    Attributes:
    -----------
    api_key : str
        The API key for accessing the Stable Diffusion API.
    api_host : str
        The host URL of the Stable Diffusion API.
    engine_id : str
        The ID of the Stable Diffusion engine.
    headers : dict
        The headers for the API request.
    output_dir : str
        Directory where generated images will be saved.

    Methods:
    --------
    generate_image(prompt: str, cfg_scale: int, height: int, width: int, samples: int, steps: int) -> List[str]:
        Generates images based on a text prompt and returns a list of file paths to the generated images.
    """

    def __init__(self, api_key: str, api_host: str = "https://api.stability.ai"):
        """
        Initializes the StableDiffusion class with the provided API key and host.

        Parameters:
        -----------
        api_key : str
            The API key for accessing the Stable Diffusion API.
        api_host : str
            The host URL of the Stable Diffusion API. Default is "https://api.stability.ai".
        """
        self.api_key = api_key
        self.api_host = api_host
        self.engine_id = "stable-diffusion-v1-6"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.output_dir = "images"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_image(self, prompt: str, cfg_scale: int = 7, height: int = 1024, width: int = 1024, samples: int = 1, steps: int = 30) -> List[str]:
        """
        Generates images based on a text prompt.

        Parameters:
        -----------
        prompt : str
            The text prompt based on which the image will be generated.
        cfg_scale : int
            CFG scale parameter for image generation. Default is 7.
        height : int
            Height of the generated image. Default is 1024.
        width : int
            Width of the generated image. Default is 1024.
        samples : int
            Number of images to generate. Default is 1.
        steps : int
            Number of steps for the generation process. Default is 30.

        Returns:
        --------
        List[str]:
            A list of paths to the generated images.

        Raises:
        -------
        Exception:
            If the API response is not 200 (OK).
        """
        response = requests.post(
            f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image",
            headers=self.headers,
            json={
                "text_prompts": [{"text": prompt}],
                "cfg_scale": cfg_scale,
                "height": height,
                "width": width,
                "samples": samples,
                "steps": steps,
            },
        )

        if response.status_code != 200:
            raise Exception(f"Non-200 response: {response.text}")

        data = response.json()
        image_paths = []
        for i, image in enumerate(data["artifacts"]):
            image_path = os.path.join(self.output_dir, f"v1_txt2img_{i}.png")
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image["base64"]))
            image_paths.append(image_path)

        return image_paths

# Usage example:
# sd = StableDiffusion("your-api-key")
# images = sd.generate_image("A scenic landscape with mountains")
# print(images)
