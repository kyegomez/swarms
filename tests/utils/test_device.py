from unittest.mock import MagicMock

import pytest
import torch

from swarms.utils.device_checker_cuda import check_device


def test_cuda_not_available(mocker):
    mocker.patch("torch.cuda.is_available", return_value=False)
    device = check_device()
    assert str(device) == "cpu"


def test_single_gpu_available(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=1)
    devices = check_device()
    assert len(devices) == 1
    assert str(devices[0]) == "cuda"


def test_multiple_gpus_available(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=2)
    devices = check_device()
    assert len(devices) == 2
    assert str(devices[0]) == "cuda:0"
    assert str(devices[1]) == "cuda:1"


def test_device_properties(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=1)
    mocker.patch(
        "torch.cuda.get_device_capability", return_value=(7, 5)
    )
    mocker.patch(
        "torch.cuda.get_device_properties",
        return_value=MagicMock(total_memory=1000),
    )
    mocker.patch("torch.cuda.memory_allocated", return_value=200)
    mocker.patch("torch.cuda.memory_reserved", return_value=300)
    mocker.patch(
        "torch.cuda.get_device_name", return_value="Tesla K80"
    )
    devices = check_device()
    assert len(devices) == 1
    assert str(devices[0]) == "cuda"


def test_memory_threshold(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=1)
    mocker.patch(
        "torch.cuda.get_device_capability", return_value=(7, 5)
    )
    mocker.patch(
        "torch.cuda.get_device_properties",
        return_value=MagicMock(total_memory=1000),
    )
    mocker.patch(
        "torch.cuda.memory_allocated", return_value=900
    )  # 90% of total memory
    mocker.patch("torch.cuda.memory_reserved", return_value=300)
    mocker.patch(
        "torch.cuda.get_device_name", return_value="Tesla K80"
    )
    with pytest.warns(
        UserWarning,
        match=r"Memory usage for device cuda exceeds threshold",
    ):
        devices = check_device(
            memory_threshold=0.8
        )  # Set memory threshold to 80%
    assert len(devices) == 1
    assert str(devices[0]) == "cuda"


def test_compute_capability_threshold(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=1)
    mocker.patch(
        "torch.cuda.get_device_capability", return_value=(3, 0)
    )  # Compute capability 3.0
    mocker.patch(
        "torch.cuda.get_device_properties",
        return_value=MagicMock(total_memory=1000),
    )
    mocker.patch("torch.cuda.memory_allocated", return_value=200)
    mocker.patch("torch.cuda.memory_reserved", return_value=300)
    mocker.patch(
        "torch.cuda.get_device_name", return_value="Tesla K80"
    )
    with pytest.warns(
        UserWarning,
        match=(
            r"Compute capability for device cuda is below threshold"
        ),
    ):
        devices = check_device(
            capability_threshold=3.5
        )  # Set compute capability threshold to 3.5
    assert len(devices) == 1
    assert str(devices[0]) == "cuda"


def test_return_single_device(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=2)
    device = check_device(return_type="single")
    assert isinstance(device, torch.device)
    assert str(device) == "cuda:0"
