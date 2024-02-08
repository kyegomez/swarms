from unittest.mock import Mock
import torch
import pytest
from swarms.models.timm import TimmModel


def test_get_supported_models():
    model_handler = TimmModel()
    supported_models = model_handler._get_supported_models()
    assert isinstance(supported_models, list)
    assert len(supported_models) > 0


def test_create_model(sample_model_info):
    model_handler = TimmModel()
    model = model_handler._create_model(sample_model_info)
    assert isinstance(model, torch.nn.Module)


def test_call(sample_model_info):
    model_handler = TimmModel()
    input_tensor = torch.randn(1, 3, 224, 224)
    output_shape = model_handler.__call__(
        sample_model_info, input_tensor
    )
    assert isinstance(output_shape, torch.Size)


def test_get_supported_models_mock():
    model_handler = TimmModel()
    model_handler._get_supported_models = Mock(
        return_value=["resnet18", "resnet50"]
    )
    supported_models = model_handler._get_supported_models()
    assert supported_models == ["resnet18", "resnet50"]


def test_create_model_mock(sample_model_info):
    model_handler = TimmModel()
    model_handler._create_model = Mock(return_value=torch.nn.Module())
    model = model_handler._create_model(sample_model_info)
    assert isinstance(model, torch.nn.Module)


def test_coverage_report():
    # Install pytest-cov
    # Run tests with coverage report
    pytest.main(["--cov=my_module", "--cov-report=html"])
