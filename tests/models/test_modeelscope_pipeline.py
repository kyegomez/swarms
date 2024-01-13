import pytest
from swarms.models.modelscope_pipeline import ModelScopePipeline
from unittest.mock import MagicMock


@pytest.fixture
def pipeline_params():
    return {
        "type_task": "text-generation",
        "model_name": "gpt2",
    }


@pytest.fixture
def pipeline_model(pipeline_params):
    return ModelScopePipeline(**pipeline_params)


def test_init(mocker, pipeline_params, pipeline_model):
    mock_pipeline = mocker.patch(
        "swarms.models.modelscope_pipeline.pipeline"
    )

    for param, value in pipeline_params.items():
        assert getattr(pipeline_model, param) == value

    mock_pipeline.assert_called_once_with(
        pipeline_params["type_task"],
        model=pipeline_params["model_name"],
    )


def test_run(mocker, pipeline_model):
    task = "Generate a 10,000 word blog on health and wellness."
    pipeline_model.model = MagicMock(return_value="Mocked output")

    output = pipeline_model.run(task)

    assert output is not None
