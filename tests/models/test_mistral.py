from unittest.mock import patch

from swarms.models.mistral import Mistral


def test_mistral_initialization():
    mistral = Mistral(device="cpu")
    assert isinstance(mistral, Mistral)
    assert mistral.ai_name == "Node Model Agent"
    assert mistral.system_prompt is None
    assert mistral.model_name == "mistralai/Mistral-7B-v0.1"
    assert mistral.device == "cpu"
    assert mistral.use_flash_attention is False
    assert mistral.temperature == 1.0
    assert mistral.max_length == 100
    assert mistral.history == []


@patch("your_module.AutoModelForCausalLM.from_pretrained")
@patch("your_module.AutoTokenizer.from_pretrained")
def test_mistral_load_model(mock_tokenizer, mock_model):
    mistral = Mistral(device="cpu")
    mistral.load_model()
    mock_model.assert_called_once()
    mock_tokenizer.assert_called_once()


@patch("your_module.Mistral.load_model")
def test_mistral_run(mock_load_model):
    mistral = Mistral(device="cpu")
    mistral.run("What's the weather in miami")
    mock_load_model.assert_called_once()


@patch("your_module.Mistral.run")
def test_mistral_chat(mock_run):
    mistral = Mistral(device="cpu")
    mistral.chat("What's the weather in miami")
    mock_run.assert_called_once()


def test_mistral__stream_response():
    mistral = Mistral(device="cpu")
    response = "It's sunny in Miami."
    tokens = list(mistral._stream_response(response))
    assert tokens == ["It's", "sunny", "in", "Miami."]
