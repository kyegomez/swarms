import os
import base64
import requests
from dotenv import load_dotenv
from typing import List

load_dotenv()

class StableDiffusion:
    def __init__(self, api_key: str, api_host: str = "https://api.stability.ai"):
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

# Example Usage
if __name__ == "__main__":
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        raise Exception("Missing Stability API key.")

    sd_api = StableDiffusion(api_key)
    images = sd_api.generate_image("A lighthouse on a cliff")
    for image_path in images:
        print(f"Generated image saved at: {image_path}")
