from typing import List

import timm
import torch
from pydantic import ConfigDict, BaseModel


class TimmModelInfo(BaseModel):
    model_name: str
    pretrained: bool
    in_chans: int
    model_config = ConfigDict(strict=True)


class TimmModel:
    """

    # Usage
    model_handler = TimmModelHandler()
    model_info = TimmModelInfo(model_name='resnet34', pretrained=True, in_chans=1)
    input_tensor = torch.randn(1, 1, 224, 224)
    output_shape = model_handler(model_info=model_info, input_tensor=input_tensor)
    print(output_shape)

    """

    def __init__(self):
        self.models = self._get_supported_models()

    def _get_supported_models(self) -> List[str]:
        """Retrieve the list of supported models from timm."""
        return timm.list_models()

    def _create_model(
        self, model_info: TimmModelInfo
    ) -> torch.nn.Module:
        """
        Create a model instance from timm with specified parameters.

        Args:
            model_info: An instance of TimmModelInfo containing model specifications.

        Returns:
            An instance of a pytorch model.
        """
        return timm.create_model(
            model_info.model_name,
            pretrained=model_info.pretrained,
            in_chans=model_info.in_chans,
        )

    def __call__(
        self, model_info: TimmModelInfo, input_tensor: torch.Tensor
    ) -> torch.Size:
        """
        Create and run a model specified by `model_info` on `input_tensor`.

        Args:
            model_info: An instance of TimmModelInfo containing model specifications.
            input_tensor: A torch tensor representing the input data.

        Returns:
            The shape of the output from the model.
        """
        model = self._create_model(model_info)
        return model(input_tensor).shape
