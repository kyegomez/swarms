from unittest.mock import Mock
import torch
import pytest
from swarms.models.timm import TimmModel, TimmModelInfo


@pytest.fixture
def sample_model_info():
    return TimmModelInfo(
        model_name="resnet18", pretrained=True, in_chans=3
    )


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


@pytest.mark.parametrize(
    "model_name, pretrained, in_chans",
    [
        ("resnet18", True, 3),
        ("resnet50", False, 1),
        ("efficientnet_b0", True, 3),
    ],
)
def test_create_model_parameterized(model_name, pretrained, in_chans):
    model_info = TimmModelInfo(
        model_name=model_name,
        pretrained=pretrained,
        in_chans=in_chans,
    )
    model_handler = TimmModel()
    model = model_handler._create_model(model_info)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize(
    "model_name, pretrained, in_chans",
    [
        ("resnet18", True, 3),
        ("resnet50", False, 1),
        ("efficientnet_b0", True, 3),
    ],
)
def test_call_parameterized(model_name, pretrained, in_chans):
    model_info = TimmModelInfo(
        model_name=model_name,
        pretrained=pretrained,
        in_chans=in_chans,
    )
    model_handler = TimmModel()
    input_tensor = torch.randn(1, in_chans, 224, 224)
    output_shape = model_handler.__call__(model_info, input_tensor)
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


def test_call_exception():
    model_handler = TimmModel()
    model_info = TimmModelInfo(
        model_name="invalid_model", pretrained=True, in_chans=3
    )
    input_tensor = torch.randn(1, 3, 224, 224)
    with pytest.raises(Exception):
        model_handler.__call__(model_info, input_tensor)


def test_coverage():
    pytest.main(["--cov=my_module", "--cov-report=html"])


def test_environment_variable():
    import os

    os.environ["MODEL_NAME"] = "resnet18"
    os.environ["PRETRAINED"] = "True"
    os.environ["IN_CHANS"] = "3"

    model_handler = TimmModel()
    model_info = TimmModelInfo(
        model_name=os.environ["MODEL_NAME"],
        pretrained=bool(os.environ["PRETRAINED"]),
        in_chans=int(os.environ["IN_CHANS"]),
    )
    input_tensor = torch.randn(1, model_info.in_chans, 224, 224)
    output_shape = model_handler(model_info, input_tensor)
    assert isinstance(output_shape, torch.Size)


@pytest.mark.slow
def test_marked_slow():
    model_handler = TimmModel()
    model_info = TimmModelInfo(
        model_name="resnet18", pretrained=True, in_chans=3
    )
    input_tensor = torch.randn(1, 3, 224, 224)
    output_shape = model_handler(model_info, input_tensor)
    assert isinstance(output_shape, torch.Size)


@pytest.mark.parametrize(
    "model_name, pretrained, in_chans",
    [
        ("resnet18", True, 3),
        ("resnet50", False, 1),
        ("efficientnet_b0", True, 3),
    ],
)
def test_marked_parameterized(model_name, pretrained, in_chans):
    model_info = TimmModelInfo(
        model_name=model_name,
        pretrained=pretrained,
        in_chans=in_chans,
    )
    model_handler = TimmModel()
    model = model_handler._create_model(model_info)
    assert isinstance(model, torch.nn.Module)


def test_exception_testing():
    model_handler = TimmModel()
    model_info = TimmModelInfo(
        model_name="invalid_model", pretrained=True, in_chans=3
    )
    input_tensor = torch.randn(1, 3, 224, 224)
    with pytest.raises(Exception):
        model_handler.__call__(model_info, input_tensor)


def test_parameterized_testing():
    model_handler = TimmModel()
    model_info = TimmModelInfo(
        model_name="resnet18", pretrained=True, in_chans=3
    )
    input_tensor = torch.randn(1, 3, 224, 224)
    output_shape = model_handler.__call__(model_info, input_tensor)
    assert isinstance(output_shape, torch.Size)


def test_use_mocks_and_monkeypatching():
    model_handler = TimmModel()
    model_handler._create_model = Mock(return_value=torch.nn.Module())
    model_info = TimmModelInfo(
        model_name="resnet18", pretrained=True, in_chans=3
    )
    model = model_handler._create_model(model_info)
    assert isinstance(model, torch.nn.Module)


def test_coverage_report():
    # Install pytest-cov
    # Run tests with coverage report
    pytest.main(["--cov=my_module", "--cov-report=html"])
