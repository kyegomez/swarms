from unittest.mock import MagicMock, patch

import pytest

from swarms.models.mixtral import Mixtral


@patch("swarms.models.mixtral.AutoTokenizer")
@patch("swarms.models.mixtral.AutoModelForCausalLM")
def test_mixtral_init(mock_model, mock_tokenizer):
    mixtral = Mixtral()
    mock_tokenizer.from_pretrained.assert_called_once()
    mock_model.from_pretrained.assert_called_once()
    assert mixtral.model_name == "mistralai/Mixtral-8x7B-v0.1"
    assert mixtral.max_new_tokens == 20


@patch("swarms.models.mixtral.AutoTokenizer")
@patch("swarms.models.mixtral.AutoModelForCausalLM")
def test_mixtral_run(mock_model, mock_tokenizer):
    mixtral = Mixtral()
    mock_tokenizer_instance = MagicMock()
    mock_model_instance = MagicMock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    mock_model.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_instance.return_tensors = "pt"
    mock_model_instance.generate.return_value = [101, 102, 103]
    mock_tokenizer_instance.decode.return_value = "Generated text"
    result = mixtral.run("Test task")
    assert result == "Generated text"
    mock_tokenizer_instance.assert_called_once_with(
        "Test task", return_tensors="pt"
    )
    mock_model_instance.generate.assert_called_once()
    mock_tokenizer_instance.decode.assert_called_once_with(
        [101, 102, 103], skip_special_tokens=True
    )


@patch("swarms.models.mixtral.AutoTokenizer")
@patch("swarms.models.mixtral.AutoModelForCausalLM")
def test_mixtral_run_error(mock_model, mock_tokenizer):
    mixtral = Mixtral()
    mock_tokenizer_instance = MagicMock()
    mock_model_instance = MagicMock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
    mock_model.from_pretrained.return_value = mock_model_instance
    mock_tokenizer_instance.return_tensors = "pt"
    mock_model_instance.generate.side_effect = Exception("Test error")
    with pytest.raises(Exception, match="Test error"):
        mixtral.run("Test task")
