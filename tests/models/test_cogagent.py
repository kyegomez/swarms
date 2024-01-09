import pytest
from swarms.models.cog_agent import CogAgent
from unittest.mock import MagicMock
from PIL import Image


@pytest.fixture
def cogagent_params():
    return {
        "model_name": "ZhipuAI/cogagent-chat",
        "tokenizer_name": "I-ModelScope/vicuna-7b-v1.5",
        "dtype": "torch.bfloat16",
        "low_cpu_mem_usage": True,
        "load_in_4bit": True,
        "trust_remote_code": True,
        "device": "cuda",
    }


@pytest.fixture
def cogagent(cogagent_params):
    return CogAgent(**cogagent_params)


def test_init(mocker, cogagent_params, cogagent):
    mock_model = mocker.patch(
        "swarms.models.cog_agent.AutoModelForCausalLM.from_pretrained"
    )
    mock_tokenizer = mocker.patch(
        "swarms.models.cog_agent.AutoTokenizer.from_pretrained"
    )

    for param, value in cogagent_params.items():
        assert getattr(cogagent, param) == value

    mock_tokenizer.assert_called_once_with(
        cogagent_params["tokenizer_name"]
    )
    mock_model.assert_called_once_with(
        cogagent_params["model_name"],
        torch_dtype=cogagent_params["dtype"],
        low_cpu_mem_usage=cogagent_params["low_cpu_mem_usage"],
        load_in_4bit=cogagent_params["load_in_4bit"],
        trust_remote_code=cogagent_params["trust_remote_code"],
    )


def test_run(mocker, cogagent):
    task = "How are you?"
    img = "images/1.jpg"
    mock_image = mocker.patch(
        "PIL.Image.open", return_value=MagicMock(spec=Image.Image)
    )
    cogagent.model.build_conversation_input_ids = MagicMock(
        return_value={
            "input_ids": MagicMock(),
            "token_type_ids": MagicMock(),
            "attention_mask": MagicMock(),
            "images": [MagicMock()],
        }
    )
    cogagent.model.__call__ = MagicMock(return_value="Mocked output")
    cogagent.decode = MagicMock(return_value="Mocked response")

    output = cogagent.run(task, img)

    assert output is not None
    mock_image.assert_called_once_with(img)
    cogagent.model.build_conversation_input_ids.assert_called_once()
    cogagent.model.__call__.assert_called_once()
    cogagent.decode.assert_called_once()
