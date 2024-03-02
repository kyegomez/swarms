# OpenAITokenizer

import pytest

import swarms.tokenizers.openai_tokenizers as tokenizers


@pytest.fixture()
def openai_tokenizer():
    return tokenizers.OpenAITokenizer("gpt-3")


def test_init(openai_tokenizer):
    assert openai_tokenizer.model == "gpt-3"


def test_default_max_tokens(openai_tokenizer):
    assert openai_tokenizer.default_max_tokens() == 4096


@pytest.mark.parametrize(
    "text, expected_output", [("Hello, world!", 3), (["Hello"], 4)]
)
def test_count_tokens_single(openai_tokenizer, text, expected_output):
    assert (
        openai_tokenizer.count_tokens(text, "gpt-3")
        == expected_output
    )


@pytest.mark.parametrize(
    "texts, expected_output",
    [(["Hello, world!", "This is a test"], 6), (["Hello"], 4)],
)
def test_count_tokens_multiple(
    openai_tokenizer, texts, expected_output
):
    assert (
        openai_tokenizer.count_tokens(texts, "gpt-3")
        == expected_output
    )


@pytest.mark.parametrize(
    "text, expected_output", [("Hello, world!", 3), (["Hello"], 4)]
)
def test_len(openai_tokenizer, text, expected_output):
    assert openai_tokenizer.len(text, "gpt-3") == expected_output
