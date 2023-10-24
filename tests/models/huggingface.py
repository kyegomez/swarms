import pytest
import torch
from unittest.mock import Mock, patch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from swarms.models.huggingface import HuggingfaceLLM


@pytest.fixture
def huggingface_llm():
    # Create an instance of HuggingfaceLLM for testing.
    model_id = "gpt2-small"
    return HuggingfaceLLM(model_id=model_id)


def test_initialization(huggingface_llm):
    # Test the initialization of the HuggingfaceLLM class.
    assert huggingface_llm.model_id == "gpt2-small"
    assert huggingface_llm.device in ["cpu", "cuda"]
    assert huggingface_llm.max_length == 20
    assert huggingface_llm.verbose == False
    assert huggingface_llm.distributed == False
    assert huggingface_llm.decoding == False
    assert huggingface_llm.model is None
    assert huggingface_llm.tokenizer is None


def test_load_model(huggingface_llm):
    # Test loading the model.
    huggingface_llm.load_model()
    assert isinstance(huggingface_llm.model, AutoModelForCausalLM)
    assert isinstance(huggingface_llm.tokenizer, AutoTokenizer)


def test_run(huggingface_llm):
    # Test the run method of HuggingfaceLLM.
    prompt_text = "Once upon a time"
    generated_text = huggingface_llm.run(prompt_text)
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0


def test_call_method(huggingface_llm):
    # Test the __call__ method of HuggingfaceLLM.
    prompt_text = "Once upon a time"
    generated_text = huggingface_llm(prompt_text)
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0


def test_load_model_failure():
    # Test loading model failure.
    with patch(
        "your_module.AutoModelForCausalLM.from_pretrained",
        side_effect=Exception("Model load failed"),
    ):
        with pytest.raises(Exception):
            huggingface_llm = HuggingfaceLLM(model_id="gpt2-small")
            huggingface_llm.load_model()
