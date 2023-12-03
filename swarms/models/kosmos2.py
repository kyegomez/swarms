from typing import List, Tuple

from PIL import Image
from pydantic import BaseModel, model_validator, validator
from transformers import AutoModelForVision2Seq, AutoProcessor


# Assuming the Detections class represents the output of the model prediction
class Detections(BaseModel):
    xyxy: List[Tuple[float, float, float, float]]
    class_id: List[int]
    confidence: List[float]

    @model_validator
    def check_length(cls, values):
        assert (
            len(values.get("xyxy"))
            == len(values.get("class_id"))
            == len(values.get("confidence"))
        ), "All fields must have the same length."
        return values

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator(
        "xyxy", "class_id", "confidence", pre=True, each_item=True
    )
    def check_not_empty(cls, v):
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("List must not be empty")
        return v

    @classmethod
    def empty(cls):
        return cls(xyxy=[], class_id=[], confidence=[])


class Kosmos2(BaseModel):
    """
    Kosmos2

    Args:
    ------
    model: AutoModelForVision2Seq
    processor: AutoProcessor

    Usage:
    ------
    >>> from swarms import Kosmos2
    >>> from swarms.models.kosmos2 import Detections
    >>> from PIL import Image
    >>> model = Kosmos2.initialize()
    >>> image = Image.open("path_to_image.jpg")
    >>> detections = model(image)
    >>> print(detections)

    """

    model: AutoModelForVision2Seq
    processor: AutoProcessor

    @classmethod
    def initialize(cls):
        model = AutoModelForVision2Seq.from_pretrained(
            "ydshieh/kosmos-2-patch14-224", trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            "ydshieh/kosmos-2-patch14-224", trust_remote_code=True
        )
        return cls(model=model, processor=processor)

    def __call__(self, img: str) -> Detections:
        image = Image.open(img)
        prompt = "<grounding>An image of"

        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        )
        outputs = self.model.generate(
            **inputs, use_cache=True, max_new_tokens=64
        )

        generated_text = self.processor.batch_decode(
            outputs, skip_special_tokens=True
        )[0]

        # The actual processing of generated_text to entities would go here
        # For the purpose of this example, assume a mock function 'extract_entities' exists:
        entities = self.extract_entities(generated_text)

        # Convert entities to detections format
        detections = self.process_entities_to_detections(
            entities, image
        )
        return detections

    def extract_entities(
        self, text: str
    ) -> List[Tuple[str, Tuple[float, float, float, float]]]:
        # Placeholder function for entity extraction
        # This should be replaced with the actual method of extracting entities
        return []

    def process_entities_to_detections(
        self,
        entities: List[Tuple[str, Tuple[float, float, float, float]]],
        image: Image.Image,
    ) -> Detections:
        if not entities:
            return Detections.empty()

        class_ids = [0] * len(
            entities
        )  # Replace with actual class ID extraction logic
        xyxys = [
            (
                e[1][0] * image.width,
                e[1][1] * image.height,
                e[1][2] * image.width,
                e[1][3] * image.height,
            )
            for e in entities
        ]
        confidences = [1.0] * len(entities)  # Placeholder confidence

        return Detections(
            xyxy=xyxys, class_id=class_ids, confidence=confidences
        )


# Usage:
# kosmos2 = Kosmos2.initialize()
# detections = kosmos2(img="path_to_image.jpg")
