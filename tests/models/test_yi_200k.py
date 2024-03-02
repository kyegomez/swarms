import pytest
import torch
from transformers import AutoTokenizer

from swarms.models.yi_200k import Yi34B200k


# Create fixtures if needed
@pytest.fixture
def yi34b_model():
    return Yi34B200k()


# Test cases for the Yi34B200k class
def test_yi34b_init(yi34b_model):
    assert isinstance(yi34b_model.model, torch.nn.Module)
    assert isinstance(yi34b_model.tokenizer, AutoTokenizer)


def test_yi34b_generate_text(yi34b_model):
    prompt = "There's a place where time stands still."
    generated_text = yi34b_model(prompt)
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0


@pytest.mark.parametrize("max_length", [64, 128, 256, 512])
def test_yi34b_generate_text_with_length(yi34b_model, max_length):
    prompt = "There's a place where time stands still."
    generated_text = yi34b_model(prompt, max_length=max_length)
    assert len(generated_text) <= max_length


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5])
def test_yi34b_generate_text_with_temperature(
    yi34b_model, temperature
):
    prompt = "There's a place where time stands still."
    generated_text = yi34b_model(prompt, temperature=temperature)
    assert isinstance(generated_text, str)


def test_yi34b_generate_text_with_invalid_prompt(yi34b_model):
    prompt = None  # Invalid prompt
    with pytest.raises(
        ValueError, match="Input prompt must be a non-empty string"
    ):
        yi34b_model(prompt)


def test_yi34b_generate_text_with_invalid_max_length(yi34b_model):
    prompt = "There's a place where time stands still."
    max_length = -1  # Invalid max_length
    with pytest.raises(
        ValueError, match="max_length must be a positive integer"
    ):
        yi34b_model(prompt, max_length=max_length)


def test_yi34b_generate_text_with_invalid_temperature(yi34b_model):
    prompt = "There's a place where time stands still."
    temperature = 2.0  # Invalid temperature
    with pytest.raises(
        ValueError, match="temperature must be between 0.01 and 1.0"
    ):
        yi34b_model(prompt, temperature=temperature)


@pytest.mark.parametrize("top_k", [20, 30, 50])
def test_yi34b_generate_text_with_top_k(yi34b_model, top_k):
    prompt = "There's a place where time stands still."
    generated_text = yi34b_model(prompt, top_k=top_k)
    assert isinstance(generated_text, str)


@pytest.mark.parametrize("top_p", [0.5, 0.7, 0.9])
def test_yi34b_generate_text_with_top_p(yi34b_model, top_p):
    prompt = "There's a place where time stands still."
    generated_text = yi34b_model(prompt, top_p=top_p)
    assert isinstance(generated_text, str)


def test_yi34b_generate_text_with_invalid_top_k(yi34b_model):
    prompt = "There's a place where time stands still."
    top_k = -1  # Invalid top_k
    with pytest.raises(
        ValueError, match="top_k must be a non-negative integer"
    ):
        yi34b_model(prompt, top_k=top_k)


def test_yi34b_generate_text_with_invalid_top_p(yi34b_model):
    prompt = "There's a place where time stands still."
    top_p = 1.5  # Invalid top_p
    with pytest.raises(
        ValueError, match="top_p must be between 0.0 and 1.0"
    ):
        yi34b_model(prompt, top_p=top_p)


@pytest.mark.parametrize("repitition_penalty", [1.0, 1.2, 1.5])
def test_yi34b_generate_text_with_repitition_penalty(
    yi34b_model, repitition_penalty
):
    prompt = "There's a place where time stands still."
    generated_text = yi34b_model(
        prompt, repitition_penalty=repitition_penalty
    )
    assert isinstance(generated_text, str)


def test_yi34b_generate_text_with_invalid_repitition_penalty(
    yi34b_model,
):
    prompt = "There's a place where time stands still."
    repitition_penalty = 0.0  # Invalid repitition_penalty
    with pytest.raises(
        ValueError,
        match="repitition_penalty must be a positive float",
    ):
        yi34b_model(prompt, repitition_penalty=repitition_penalty)


def test_yi34b_generate_text_with_invalid_device(yi34b_model):
    prompt = "There's a place where time stands still."
    device_map = "invalid_device"  # Invalid device_map
    with pytest.raises(ValueError, match="Invalid device_map"):
        yi34b_model(prompt, device_map=device_map)
