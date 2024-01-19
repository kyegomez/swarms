from typing import List

import timm
import torch
from torch import Tensor
from swarms.models.base_multimodal_model import BaseMultiModalModel


class TimmModel(BaseMultiModalModel):
    """
    TimmModel is a class that wraps the timm library to provide a consistent
    interface for creating and running models.

    Args:
        model_name: A string representing the name of the model to be created.
        pretrained: A boolean indicating whether to use a pretrained model.
        in_chans: An integer representing the number of input channels.

    Returns:
        A TimmModel instance.

    Example:
        model = TimmModel('resnet18', pretrained=True, in_chans=3)
        output_shape = model(input_tensor)

    """

    def __init__(
        self, model_name: str, pretrained: bool, in_chans: int
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.in_chans = in_chans
        self.models = self._get_supported_models()

    def _get_supported_models(self) -> List[str]:
        """Retrieve the list of supported models from timm."""
        return timm.list_models()

    def __call__(self, task: Tensor, *args, **kwargs) -> torch.Size:
        """
        Create and run a model specified by `model_info` on `input_tensor`.

        Args:
            model_info: An instance of TimmModelInfo containing model specifications.
            input_tensor: A torch tensor representing the input data.

        Returns:
            The shape of the output from the model.
        """
        model = timm.create_model(self.model_name, *args, **kwargs)
        return model(task)

    def list_models(self):
        return timm.list_models()
