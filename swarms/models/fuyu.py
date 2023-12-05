from io import BytesIO

import requests
from PIL import Image
from transformers import (
    AutoTokenizer,
    FuyuForCausalLM,
    FuyuImageProcessor,
    FuyuProcessor,
)


class Fuyu:
    """
    Fuyu model by Adept


    Parameters
    ----------
    pretrained_path : str
        Path to the pretrained model
    device_map : str
        Device to use for the model
    max_new_tokens : int
        Maximum number of tokens to generate

    Examples
    --------
    >>> fuyu = Fuyu()
    >>> fuyu("Hello, my name is", "path/to/image.png")

    """

    def __init__(
        self,
        pretrained_path: str = "adept/fuyu-8b",
        device_map: str = "auto",
        max_new_tokens: int = 500,
        *args,
        **kwargs,
    ):
        self.pretrained_path = pretrained_path
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_path
        )
        self.image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            **kwargs,
        )
        self.model = FuyuForCausalLM.from_pretrained(
            pretrained_path,
            device_map=device_map,
            **kwargs,
        )

    def get_img(self, img: str):
        """Get the image from the path"""
        image_pil = Image.open(img)
        return image_pil

    def __call__(self, text: str, img: str):
        """Call the model with text and img paths"""
        img = self.get_img(img)
        model_inputs = self.processor(
            text=text, images=[img], device=self.device_map
        )

        for k, v in model_inputs.items():
            model_inputs[k] = v.to(self.device_map)

        output = self.model.generate(
            **model_inputs, max_new_tokens=self.max_new_tokens
        )
        text = self.processor.batch_decode(
            output[:, -7:], skip_special_tokens=True
        )
        return print(str(text))

    def get_img_from_web(self, img: str):
        """Get the image from the web"""
        try:
            response = requests.get(img)
            response.raise_for_status()
            image_pil = Image.open(BytesIO(response.content))
            return image_pil
        except requests.RequestException as error:
            print(
                f"Error fetching image from {img} and error: {error}"
            )
            return None
