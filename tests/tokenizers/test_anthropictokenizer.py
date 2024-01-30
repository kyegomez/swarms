# AnthropicTokenizer

import pytest
from swarms.tokenizers.anthropic_tokenizer import AnthropicTokenizer


def test_post_init():
    tokenizer = AnthropicTokenizer()
    assert tokenizer.model == "claude-2.1"
    assert tokenizer.max_tokens == 200000


def test_default_max_tokens():
    tokenizer = AnthropicTokenizer(model="claude")
    assert tokenizer.default_max_tokens() == 100000


@pytest.mark.parametrize(
    "model,tokens", [("claude-2.1", 200000), ("claude", 100000)]
)
def test_default_max_tokens_models(model, tokens):
    tokenizer = AnthropicTokenizer(model=model)
    assert tokenizer.default_max_tokens() == tokens


def test_count_tokens_string():
    # Insert mock instantiation of anthropic client and its count_tokens function
    text = "This is a test string."
    tokenizer = AnthropicTokenizer()
    tokens = tokenizer.count_tokens(text)
    assert tokens == 5


def test_count_tokens_list():
    # Insert mock instantiation of anthropic client and its count_tokens function
    text = ["This", "is", "a", "test", "string."]
    tokenizer = AnthropicTokenizer()
    with pytest.raises(ValueError):
        tokenizer.count_tokens(text)
