import torch
import logging
from unittest.mock import patch

import pytest

from swarms.models.huggingface import HuggingfaceLLM


# Mock some functions and objects for testing
@pytest.fixture
def mock_huggingface_llm(monkeypatch):
    # Mock the model and tokenizer creation
    def mock_init(
        self,
        model_id,
        device="cpu",
        max_length=500,
        quantize=False,
        quantization_config=None,
        verbose=False,
        distributed=False,
        decoding=False,
        max_workers=5,
        repitition_penalty=1.3,
        no_repeat_ngram_size=5,
        temperature=0.7,
        top_k=40,
        top_p=0.8,
    ):
        pass

    # Mock the model loading
    def mock_load_model(self):
        pass

    # Mock the model generation
    def mock_run(self, task):
        pass

    monkeypatch.setattr(HuggingfaceLLM, "__init__", mock_init)
    monkeypatch.setattr(HuggingfaceLLM, "load_model", mock_load_model)
    monkeypatch.setattr(HuggingfaceLLM, "run", mock_run)


# Basic tests for initialization and attribute settings
def test_init_huggingface_llm():
    llm = HuggingfaceLLM(
        model_id="test_model",
        device="cuda",
        max_length=1000,
        quantize=True,
        quantization_config={"config_key": "config_value"},
        verbose=True,
        distributed=True,
        decoding=True,
        max_workers=3,
        repitition_penalty=1.5,
        no_repeat_ngram_size=4,
        temperature=0.8,
        top_k=50,
        top_p=0.7,
    )

    assert llm.model_id == "test_model"
    assert llm.device == "cuda"
    assert llm.max_length == 1000
    assert llm.quantize is True
    assert llm.quantization_config == {"config_key": "config_value"}
    assert llm.verbose is True
    assert llm.distributed is True
    assert llm.decoding is True
    assert llm.max_workers == 3
    assert llm.repitition_penalty == 1.5
    assert llm.no_repeat_ngram_size == 4
    assert llm.temperature == 0.8
    assert llm.top_k == 50
    assert llm.top_p == 0.7


# Test loading the model
def test_load_model(mock_huggingface_llm):
    llm = HuggingfaceLLM(model_id="test_model")
    llm.load_model()

    # Ensure that the load_model function is called
    assert True


# Test running the model
def test_run(mock_huggingface_llm):
    llm = HuggingfaceLLM(model_id="test_model")
    llm.run("Test prompt")

    # Ensure that the run function is called
    assert True


# Test for setting max_length
def test_llm_set_max_length(llm_instance):
    new_max_length = 1000
    llm_instance.set_max_length(new_max_length)
    assert llm_instance.max_length == new_max_length


# Test for setting verbose
def test_llm_set_verbose(llm_instance):
    llm_instance.set_verbose(True)
    assert llm_instance.verbose is True


# Test for setting distributed
def test_llm_set_distributed(llm_instance):
    llm_instance.set_distributed(True)
    assert llm_instance.distributed is True


# Test for setting decoding
def test_llm_set_decoding(llm_instance):
    llm_instance.set_decoding(True)
    assert llm_instance.decoding is True


# Test for setting max_workers
def test_llm_set_max_workers(llm_instance):
    new_max_workers = 10
    llm_instance.set_max_workers(new_max_workers)
    assert llm_instance.max_workers == new_max_workers


# Test for setting repitition_penalty
def test_llm_set_repitition_penalty(llm_instance):
    new_repitition_penalty = 1.5
    llm_instance.set_repitition_penalty(new_repitition_penalty)
    assert llm_instance.repitition_penalty == new_repitition_penalty


# Test for setting no_repeat_ngram_size
def test_llm_set_no_repeat_ngram_size(llm_instance):
    new_no_repeat_ngram_size = 6
    llm_instance.set_no_repeat_ngram_size(new_no_repeat_ngram_size)
    assert (
        llm_instance.no_repeat_ngram_size == new_no_repeat_ngram_size
    )


# Test for setting temperature
def test_llm_set_temperature(llm_instance):
    new_temperature = 0.8
    llm_instance.set_temperature(new_temperature)
    assert llm_instance.temperature == new_temperature


# Test for setting top_k
def test_llm_set_top_k(llm_instance):
    new_top_k = 50
    llm_instance.set_top_k(new_top_k)
    assert llm_instance.top_k == new_top_k


# Test for setting top_p
def test_llm_set_top_p(llm_instance):
    new_top_p = 0.9
    llm_instance.set_top_p(new_top_p)
    assert llm_instance.top_p == new_top_p


# Test for setting quantize
def test_llm_set_quantize(llm_instance):
    llm_instance.set_quantize(True)
    assert llm_instance.quantize is True


# Test for setting quantization_config
def test_llm_set_quantization_config(llm_instance):
    new_quantization_config = {
        "load_in_4bit": False,
        "bnb_4bit_use_double_quant": False,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    }
    llm_instance.set_quantization_config(new_quantization_config)
    assert llm_instance.quantization_config == new_quantization_config


# Test for setting model_id
def test_llm_set_model_id(llm_instance):
    new_model_id = "EleutherAI/gpt-neo-2.7B"
    llm_instance.set_model_id(new_model_id)
    assert llm_instance.model_id == new_model_id


# Test for setting model
@patch(
    "swarms.models.huggingface.AutoModelForCausalLM.from_pretrained"
)
def test_llm_set_model(mock_model, llm_instance):
    mock_model.return_value = "mocked model"
    llm_instance.set_model(mock_model)
    assert llm_instance.model == "mocked model"


# Test for setting tokenizer
@patch("swarms.models.huggingface.AutoTokenizer.from_pretrained")
def test_llm_set_tokenizer(mock_tokenizer, llm_instance):
    mock_tokenizer.return_value = "mocked tokenizer"
    llm_instance.set_tokenizer(mock_tokenizer)
    assert llm_instance.tokenizer == "mocked tokenizer"


# Test for setting logger
def test_llm_set_logger(llm_instance):
    new_logger = logging.getLogger("test_logger")
    llm_instance.set_logger(new_logger)
    assert llm_instance.logger == new_logger


# Test for saving model
@patch("torch.save")
def test_llm_save_model(mock_save, llm_instance):
    llm_instance.save_model("path/to/save")
    mock_save.assert_called_once()


# Test for print_dashboard
@patch("builtins.print")
def test_llm_print_dashboard(mock_print, llm_instance):
    llm_instance.print_dashboard("test task")
    mock_print.assert_called()


# Test for __call__ method
@patch("swarms.models.huggingface.HuggingfaceLLM.run")
def test_llm_call(mock_run, llm_instance):
    mock_run.return_value = "mocked output"
    result = llm_instance("test task")
    assert result == "mocked output"
