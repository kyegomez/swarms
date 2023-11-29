import pytest
import torch
from unittest.mock import Mock
from swarms.models.huggingface import HuggingFaceLLM


@pytest.fixture
def mock_torch():
    return Mock()


@pytest.fixture
def mock_autotokenizer():
    return Mock()


@pytest.fixture
def mock_automodelforcausallm():
    return Mock()


@pytest.fixture
def mock_bitsandbytesconfig():
    return Mock()


@pytest.fixture
def hugging_face_llm(
    mock_torch,
    mock_autotokenizer,
    mock_automodelforcausallm,
    mock_bitsandbytesconfig,
):
    HuggingFaceLLM.torch = mock_torch
    HuggingFaceLLM.AutoTokenizer = mock_autotokenizer
    HuggingFaceLLM.AutoModelForCausalLM = mock_automodelforcausallm
    HuggingFaceLLM.BitsAndBytesConfig = mock_bitsandbytesconfig

    return HuggingFaceLLM(model_id="test")


def test_init(
    hugging_face_llm, mock_autotokenizer, mock_automodelforcausallm
):
    assert hugging_face_llm.model_id == "test"
    mock_autotokenizer.from_pretrained.assert_called_once_with("test")
    mock_automodelforcausallm.from_pretrained.assert_called_once_with(
        "test", quantization_config=None
    )


def test_init_with_quantize(
    hugging_face_llm,
    mock_autotokenizer,
    mock_automodelforcausallm,
    mock_bitsandbytesconfig,
):
    quantization_config = {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    }
    mock_bitsandbytesconfig.return_value = quantization_config

    HuggingFaceLLM(model_id="test", quantize=True)

    mock_bitsandbytesconfig.assert_called_once_with(
        **quantization_config
    )
    mock_autotokenizer.from_pretrained.assert_called_once_with("test")
    mock_automodelforcausallm.from_pretrained.assert_called_once_with(
        "test", quantization_config=quantization_config
    )


def test_generate_text(hugging_face_llm):
    prompt_text = "test prompt"
    expected_output = "test output"
    hugging_face_llm.tokenizer.encode.return_value = torch.tensor(
        [0]
    )  # Mock tensor
    hugging_face_llm.model.generate.return_value = torch.tensor(
        [0]
    )  # Mock tensor
    hugging_face_llm.tokenizer.decode.return_value = expected_output

    output = hugging_face_llm.generate_text(prompt_text)

    assert output == expected_output
