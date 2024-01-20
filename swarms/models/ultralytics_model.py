from swarms.models.base_multimodal_model import BaseMultiModalModel
from ultralytics import YOLO


class UltralyticsModel(BaseMultiModalModel):
    """
    Initializes an instance of the Ultralytics model.

    Args:
        model_name (str): The name of the model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

        self.model = YOLO(model_name, *args, **kwargs)

    def __call__(self, task: str, *args, **kwargs):
        """
        Calls the Ultralytics model.

        Args:
            task (str): The task to perform.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of the model call.
        """
        return self.model(task, *args, **kwargs)
