from typing import Union

from roboflow import Roboflow

from swarms.models.base_multimodal_model import BaseMultiModalModel


class RoboflowMultiModal(BaseMultiModalModel):
    """
    Initializes the RoboflowModel with the given API key, project ID, and version.

    Args:
        api_key (str): The API key for Roboflow.
        project_id (str): The ID of the project.
        version (str): The version of the model.
        confidence (int, optional): The confidence threshold. Defaults to 50.
        overlap (int, optional): The overlap threshold. Defaults to 25.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        version: str,
        confidence: int = 50,
        overlap: int = 25,
        hosted: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.api_key = api_key
        self.project_id = project_id
        self.verison = version
        self.confidence = confidence
        self.overlap = overlap
        self.hosted = hosted

        try:
            rf = Roboflow(api_key=api_key, *args, **kwargs)
            project = rf.workspace().project(project_id)
            self.model = project.version(version).model
            self.model.confidence = confidence
            self.model.overlap = overlap
        except Exception as e:
            print(f"Error initializing RoboflowModel: {str(e)}")

    def __call__(self, img: Union[str, bytes]):
        """
        Runs inference on an image and retrieves predictions.

        Args:
            img (Union[str, bytes]): The path to the image or the URL of the image.
            hosted (bool, optional): Whether the image is hosted. Defaults to False.

        Returns:
            Optional[roboflow.Prediction]: The prediction or None if an error occurs.
        """
        try:
            prediction = self.model.predict(img, hosted=self.hosted)
            return prediction
        except Exception as e:
            print(f"Error running inference: {str(e)}")
            return None
