import pytest
import torch
from unittest.mock import MagicMock
from swarms.utils.load_model_torch import load_model_torch


def test_load_model_torch_no_model_path():
    with pytest.raises(FileNotFoundError):
        load_model_torch()


def test_load_model_torch_model_not_found(mocker):
    mocker.patch("torch.load", side_effect=FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        load_model_torch("non_existent_model_path")


def test_load_model_torch_runtime_error(mocker):
    mocker.patch("torch.load", side_effect=RuntimeError)
    with pytest.raises(RuntimeError):
        load_model_torch("model_path")


def test_load_model_torch_no_device_specified(mocker):
    mock_model = MagicMock(spec=torch.nn.Module)
    mocker.patch("torch.load", return_value=mock_model)
    mocker.patch("torch.cuda.is_available", return_value=False)
    model = load_model_torch("model_path")
    mock_model.to.assert_called_once_with(torch.device("cpu"))


def test_load_model_torch_device_specified(mocker):
    mock_model = MagicMock(spec=torch.nn.Module)
    mocker.patch("torch.load", return_value=mock_model)
    model = load_model_torch(
        "model_path", device=torch.device("cuda")
    )
    mock_model.to.assert_called_once_with(torch.device("cuda"))


def test_load_model_torch_model_specified(mocker):
    mock_model = MagicMock(spec=torch.nn.Module)
    mocker.patch("torch.load", return_value={"key": "value"})
    load_model_torch("model_path", model=mock_model)
    mock_model.load_state_dict.assert_called_once_with(
        {"key": "value"}, strict=True
    )


def test_load_model_torch_model_specified_strict_false(mocker):
    mock_model = MagicMock(spec=torch.nn.Module)
    mocker.patch("torch.load", return_value={"key": "value"})
    load_model_torch("model_path", model=mock_model, strict=False)
    mock_model.load_state_dict.assert_called_once_with(
        {"key": "value"}, strict=False
    )
