from typing import Optional, Callable
import cv2
import numpy as np
import supervision as sv
from PIL import Image
from transformers import (
    SamImageProcessor,
    SamModel,
    SamProcessor,
    pipeline,
)

from swarms.models.base_multimodal_model import BaseMultiModalModel


class SegmentAnythingMarkGenerator(BaseMultiModalModel):
    """
    A class for performing image segmentation using a specified model.

    Parameters:
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        model_name (str): The name of the model to be loaded. Defaults to
            'facebook/sam-vit-huge'.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "facebook/sam-vit-huge",
        visualize_marks: bool = False,
        masks_to_marks: Callable = sv.masks_to_marks,
        *args,
        **kwargs,
    ):
        super(SegmentAnythingMarkGenerator).__init__(*args, **kwargs)
        self.device = device
        self.model_name = model_name
        self.visualize_marks = visualize_marks
        self.masks_to_marks = masks_to_marks

        self.model = SamModel.from_pretrained(
            model_name, *args, **kwargs
        ).to(device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.image_processor = SamImageProcessor.from_pretrained(
            model_name
        )
        self.device = device
        self.pipeline = pipeline(
            task="mask-generation",
            model=self.model,
            image_processor=self.image_processor,
            device=self.device,
        )

    def __call__(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> sv.Detections:
        """
        Generate image segmentation marks.

        Parameters:
            image (np.ndarray): The image to be marked in BGR format.
            mask: (Optional[np.ndarray]): The mask to be used as a guide for
                segmentation.

        Returns:
            sv.Detections: An object containing the segmentation masks and their
                corresponding bounding box coordinates.
        """
        image = Image.fromarray(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        if mask is None:
            outputs = self.pipeline(image, points_per_batch=64)
            masks = np.array(outputs["masks"])
            return self.masks_to_marks(masks=masks)
        else:
            inputs = self.processor(image, return_tensors="pt").to(
                self.device
            )
            image_embeddings = self.model.get_image_embeddings(
                inputs.pixel_values
            )
            masks = []
            for polygon in sv.mask_to_polygons(mask.astype(bool)):
                indexes = np.random.choice(
                    a=polygon.shape[0], size=5, replace=True
                )
                input_points = polygon[indexes]
                inputs = self.processor(
                    images=image,
                    input_points=[[input_points]],
                    return_tensors="pt",
                ).to(self.device)
                del inputs["pixel_values"]
                outputs = self.model(
                    image_embeddings=image_embeddings, **inputs
                )
                mask = (
                    self.processor.image_processor.post_process_masks(
                        masks=outputs.pred_masks.cpu().detach(),
                        original_sizes=inputs["original_sizes"]
                        .cpu()
                        .detach(),
                        reshaped_input_sizes=inputs[
                            "reshaped_input_sizes"
                        ]
                        .cpu()
                        .detach(),
                    )[0][0][0].numpy()
                )
                masks.append(mask)
            masks = np.array(masks)
            return self.masks_to_marks(masks=masks)

    # def visualize_img(self):
