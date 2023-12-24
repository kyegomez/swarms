import torch
from unittest.mock import MagicMock
from swarms.utils.prep_torch_model_inference import (
    prep_torch_inference,
)


def test_prep_torch_inference_no_model_path():
    result = prep_torch_inference()
    assert result is None


def test_prep_torch_inference_model_not_found(mocker):
    mocker.patch(
        "swarms.utils.prep_torch_model_inference.load_model_torch",
        side_effect=FileNotFoundError,
    )
    result = prep_torch_inference("non_existent_model_path")
    assert result is None


def test_prep_torch_inference_runtime_error(mocker):
    mocker.patch(
        "swarms.utils.prep_torch_model_inference.load_model_torch",
        side_effect=RuntimeError,
    )
    result = prep_torch_inference("model_path")
    assert result is None


def test_prep_torch_inference_no_device_specified(mocker):
    mock_model = MagicMock(spec=torch.nn.Module)
    mocker.patch(
        "swarms.utils.prep_torch_model_inference.load_model_torch",
        return_value=mock_model,
    )
    prep_torch_inference("model_path")
    mock_model.eval.assert_called_once()


def test_prep_torch_inference_device_specified(mocker):
    mock_model = MagicMock(spec=torch.nn.Module)
    mocker.patch(
        "swarms.utils.prep_torch_model_inference.load_model_torch",
        return_value=mock_model,
    )
    prep_torch_inference("model_path", device=torch.device("cuda"))
    mock_model.eval.assert_called_once()
