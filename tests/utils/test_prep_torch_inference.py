import unittest
import pytest
import torch
from unittest.mock import Mock
from swarms.utils import prep_torch_inference


def test_prep_torch_inference():
    model_path = "model_path"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model_mock = Mock()
    model_mock.eval = Mock()

    # Mocking the load_model_torch function to return our mock model.
    with unittest.mock.patch(
        "swarms.utils.load_model_torch", return_value=model_mock
    ) as _:
        model = prep_torch_inference(model_path, device)

    # Check if model was properly loaded and eval function was called
    assert model == model_mock
    model_mock.eval.assert_called_once()


@pytest.mark.parametrize(
    "model_path, device",
    [
        (
            "invalid_path",
            torch.device("cuda"),
        ),  # Invalid file path, valid device
        (None, torch.device("cuda")),  # None file path, valid device
        ("model_path", None),  # Valid file path, None device
        (None, None),  # None file path, None device
    ],
)
def test_prep_torch_inference_exceptions(model_path, device):
    with pytest.raises(Exception):
        prep_torch_inference(model_path, device)


def test_prep_torch_inference_return_none():
    model_path = "invalid_path"  # Invalid file path
    device = torch.device("cuda")  # Valid device

    # Since load_model_torch function will raise an exception, prep_torch_inference should return None
    assert prep_torch_inference(model_path, device) is None
