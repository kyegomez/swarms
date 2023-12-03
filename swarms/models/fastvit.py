import json
import os
from typing import List

import timm
import torch
from PIL import Image
from pydantic import BaseModel, StrictFloat, StrictInt, validator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the classes for image classification
with open(
    os.path.join(os.path.dirname(__file__), "fast_vit_classes.json")
) as f:
    FASTVIT_IMAGENET_1K_CLASSES = json.load(f)


class ClassificationResult(BaseModel):
    class_id: List[StrictInt]
    confidence: List[StrictFloat]

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("class_id", "confidence", pre=True, each_item=True)
    def check_list_contents(cls, v):
        assert isinstance(v, int) or isinstance(
            v, float
        ), "must be integer or float"
        return v


class FastViT:
    """
    FastViT model for image classification

    Args:
        img (str): path to the input image
        confidence_threshold (float): confidence threshold for the model's predictions

    Returns:
        ClassificationResult: a pydantic BaseModel containing the class ids and confidences of the model's predictions


    Example:
        >>> fastvit = FastViT()
        >>> result = fastvit(img="path_to_image.jpg", confidence_threshold=0.5)


    To use, create a json file called: fast_vit_classes.json

    """

    def __init__(self):
        self.model = timm.create_model(
            "hf_hub:timm/fastvit_s12.apple_in1k", pretrained=True
        ).to(DEVICE)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **data_config, is_training=False
        )
        self.model.eval()

    def __call__(
        self, img: str, confidence_threshold: float = 0.5
    ) -> ClassificationResult:
        """classifies the input image and returns the top k classes and their probabilities"""
        img = Image.open(img).convert("RGB")
        img_tensor = self.transforms(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Get top k classes and their probabilities
        top_probs, top_classes = torch.topk(
            probabilities, k=FASTVIT_IMAGENET_1K_CLASSES
        )

        # Filter by confidence threshold
        mask = top_probs > confidence_threshold
        top_probs, top_classes = top_probs[mask], top_classes[mask]

        # Convert to Python lists and map class indices to labels if needed
        top_probs = top_probs.cpu().numpy().tolist()
        top_classes = top_classes.cpu().numpy().tolist()
        # top_class_labels = [FASTVIT_IMAGENET_1K_CLASSES[i] for i in top_classes] # Uncomment if class labels are needed

        return ClassificationResult(
            class_id=top_classes, confidence=top_probs
        )
