from typing import Any

import cv2

from swarms.models.base_multimodal_model import BaseMultiModalModel
from swarms.models.sam_supervision import SegmentAnythingMarkGenerator
from swarms.utils.supervision_masking import refine_marks
from swarms.utils.supervision_visualizer import MarkVisualizer


class GPT4VSAM(BaseMultiModalModel):
    """
    GPT4VSAM class represents a multi-modal model that combines the capabilities of GPT-4 and SegmentAnythingMarkGenerator.
    It takes an instance of BaseMultiModalModel (vlm)
    and a device as input and provides methods for loading images and making predictions.

    Args:
        vlm (BaseMultiModalModel): An instance of BaseMultiModalModel representing the visual language model.
        device (str, optional): The device to be used for computation. Defaults to "cuda".

    Attributes:
        vlm (BaseMultiModalModel): An instance of BaseMultiModalModel representing the visual language model.
        device (str): The device to be used for computation.
        sam (SegmentAnythingMarkGenerator): An instance of SegmentAnythingMarkGenerator for generating marks.
        visualizer (MarkVisualizer): An instance of MarkVisualizer for visualizing marks.

    Methods:
        load_img(img: str) -> Any: Loads an image from the given file path.
        __call__(task: str, img: str, *args, **kwargs) -> Any: Makes predictions using the visual language model.

    """

    def __init__(
        self,
        vlm: BaseMultiModalModel,
        device: str = "cuda",
        return_related_marks: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vlm = vlm
        self.device = device
        self.return_related_marks = return_related_marks

        self.sam = SegmentAnythingMarkGenerator(
            device, *args, **kwargs
        )
        self.visualizer = MarkVisualizer(*args, **kwargs)

    def load_img(self, img: str) -> Any:
        """
        Loads an image from the given file path.

        Args:
            img (str): The file path of the image.

        Returns:
            Any: The loaded image.

        """
        return cv2.imread(img)

    def __call__(self, task: str, img: str, *args, **kwargs) -> Any:
        """
        Makes predictions using the visual language model.

        Args:
            task (str): The task for which predictions are to be made.
            img (str): The file path of the image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The predictions made by the visual language model.

        """
        img = self.load_img(img)

        marks = self.sam(image=img)
        marks = refine_marks(marks=marks)

        return self.vlm(task, img, *args, **kwargs)
