import pytest
from swarms.models.modelscope_llm import ModelScopeAutoModel
from unittest.mock import MagicMock


@pytest.fixture
def model_params():
    return {
        "model_name": "gpt2",
        "tokenizer_name": None,
        "device": "cuda",
        "device_map": "auto",
        "max_new_tokens": 500,
        "skip_special_tokens": True,
    }


@pytest.fixture
def modelscope(model_params):
    return ModelScopeAutoModel(**model_params)


def test_init(mocker, model_params, modelscope):
    mock_model = mocker.patch(
        "swarms.models.modelscope_llm.AutoModelForCausalLM.from_pretrained"
    )
    mock_tokenizer = mocker.patch(
        "swarms.models.modelscope_llm.AutoTokenizer.from_pretrained"
    )

    for param, value in model_params.items():
        assert getattr(modelscope, param) == value

    mock_tokenizer.assert_called_once_with(
        model_params["tokenizer_name"]
    )
    mock_model.assert_called_once_with(
        model_params["model_name"],
        device_map=model_params["device_map"],
    )


def test_run(mocker, modelscope):
    task = "Generate a 10,000 word blog on health and wellness."
    mocker.patch(
        "swarms.models.modelscope_llm.AutoTokenizer.decode",
        return_value="Mocked output",
    )
    modelscope.model.generate = MagicMock(
        return_value=["Mocked token"]
    )
    modelscope.tokenizer = MagicMock(
        return_value={"input_ids": "Mocked input_ids"}
    )

    output = modelscope.run(task)

    assert output is not None
