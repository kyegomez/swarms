from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image


class Vilt:
    """
    Vision-and-Language Transformer (ViLT) model fine-tuned on VQAv2.
    It was introduced in the paper ViLT: Vision-and-Language Transformer Without
    Convolution or Region Supervision by Kim et al. and first released in this repository.

    Disclaimer: The team releasing ViLT did not write a model card for this model
    so this model card has been written by the Hugging Face team.

    https://huggingface.co/dandelin/vilt-b32-finetuned-vqa


    Example:
        >>> model = Vilt()
        >>> output = model("What is this image", "http://images.cocodataset.org/val2017/000000039769.jpg")

    """

    def __init__(self):
        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        self.model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )

    def __call__(self, text: str, image_url: str):
        """
        Run the model


        Args:

        """
        # Download the image
        image = Image.open(requests.get(image_url, stream=True).raw)

        encoding = self.processor(image, text, return_tensors="pt")

        # Forward pass
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        print("Predicted Answer:", self.model.config.id2label[idx])
