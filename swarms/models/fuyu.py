"""Fuyu model by Kye"""
from transformers import (
    FuyuForCausalLM,
    AutoTokenizer,
    FuyuProcessor,
    FuyuImageProcessor,
)
from PIL import Image


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
        device_map: str = "cuda:0",
        max_new_tokens: int = 7,
    ):
        self.pretrained_path = pretrained_path
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(
            image_procesor=self.image_processor, tokenizer=self.tokenizer
        )
        self.model = FuyuForCausalLM.from_pretrained(
            pretrained_path, device_map=device_map
        )

    def __call__(self, text: str, img_path: str):
        """Call the model with text and img paths"""
        image_pil = Image.open(img_path)
        model_inputs = self.processor(
            text=text, images=[image_pil], device=self.device_map
        )

        for k, v in model_inputs.items():
            model_inputs[k] = v.to(self.device_map)

        output = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
        text = self.processor.batch_decode(output[:, -7:], skip_special_tokens=True)
