import io
import requests
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)
from swarms.models.base_multimodal_model import (
    BaseMultiModalModel,
)  # noqa: F401


class ChestMultiModalAgent(BaseMultiModalModel):
    """
    Initialize the ChestAgent.

    Args:
        device (str): The device to run the model on. Default is "cuda".
        dtype (torch.dtype): The data type to use for the model. Default is torch.float16.
        model_name (str): The name or path of the pre-trained model to use. Default is "StanfordAIMI/CheXagent-8b".

    Example:
        >>> agent = ChestAgent()
        >>> agent.run("What are the symptoms of COVID-19?", "https://example.com/image.jpg")

    """

    def __init__(
        self,
        device="cuda",
        dtype=torch.float16,
        model_name="StanfordAIMI/CheXagent-8b",
        *args,
        **kwargs,
    ):
        # Step 1: Setup constants
        self.device = device
        self.dtype = dtype

        # Step 2: Load Processor and Model
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.generation_config = GenerationConfig.from_pretrained(
            model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            *args,
            **kwargs,
        )

    def run(self, task: str, img: str, *args, **kwargs):
        """
        Run the ChestAgent to generate findings based on an image and a prompt.

        Args:
            image_path (str): The URL or local path of the image.
            prompt (str): The prompt to use for generating findings.

        Returns:
            str: The generated findings.
        """
        # Step 3: Fetch the images
        images = [
            Image.open(io.BytesIO(requests.get(img).content)).convert(
                "RGB"
            )
        ]

        # Step 4: Generate the Findings section
        inputs = self.processor(
            images=images,
            text=f" USER: <s>{task} ASSISTANT: <s>",
            return_tensors="pt",
        ).to(device=self.device, dtype=self.dtype)
        output = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
        )[0]
        response = self.processor.tokenizer.decode(
            output, skip_special_tokens=True
        )

        return response
