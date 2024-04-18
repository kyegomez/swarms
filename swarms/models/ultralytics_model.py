from typing import List

from ultralytics import YOLO

from swarms.models.base_multimodal_model import BaseMultiModalModel


class UltralyticsModel(BaseMultiModalModel):
    """
    Initializes an instance of the Ultralytics model.

    Args:
        model_name (str): The name of the model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, model_name: str = "yolov8n.pt", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

        try:
            self.model = YOLO(model_name, *args, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Ultralytics model: {str(e)}"
            )

    def __call__(
        self, task: str, tasks: List[str] = None, *args, **kwargs
    ):
        """
        Calls the Ultralytics model.

        Args:
            task (str): The task to perform.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of the model call.
        """
        try:
            if tasks:
                return self.model([tasks], *args, **kwargs)
            else:
                return self.model(task, *args, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to perform task '{task}' with Ultralytics"
                f" model: {str(e)}"
            )
