from typing import Any
from litellm import image_generation


class ImageGenerator:
    def __init__(
        self,
        model: str | None = None,
        n: int | None = 2,
        quality: Any = None,
        response_format: str | None = None,
        size: str | None = 10,
        style: str | None = None,
        user: str | None = None,
        input_fidelity: str | None = None,
        timeout: int = 600,
        output_path_folder: str | None = "images",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model = model
        self.n = n
        self.quality = quality
        self.response_format = response_format
        self.size = size
        self.style = style
        self.user = user
        self.input_fidelity = input_fidelity
        self.timeout = timeout
        self.output_path_folder = output_path_folder
        self.api_key = api_key
        self.api_base = api_base

    def run(self, task: str = None):

        return image_generation(
            prompt=task,
            model=self.model,
            n=self.n,
            quality=self.quality,
            response_format=self.response_format,
            size=self.size,
            style=self.style,
            user=self.user,
            input_fidelity=self.input_fidelity,
            timeout=self.timeout,
        )


# if __name__ == "__main__":
#     image_generator = ImageGenerator()
#     print(image_generator.run(task="A beautiful sunset over a calm ocean"))

# print(model_list)
