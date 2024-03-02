# BaseTokenizer

import pytest

from swarms.tokenizers.base_tokenizer import BaseTokenizer


# 1. Fixture for BaseTokenizer instance.
@pytest.fixture
def base_tokenizer():
    return BaseTokenizer(max_tokens=100)


# 2. Tests for __post_init__.
def test_post_init(base_tokenizer):
    assert base_tokenizer.stop_sequences == ["<|Response|>"]
    assert base_tokenizer.stop_token == "<|Response|>"


# 3. Tests for count_tokens_left with different inputs.
def test_count_tokens_left_with_positive_diff(
    base_tokenizer, monkeypatch
):
    # Mocking count_tokens to return a specific value
    monkeypatch.setattr(
        "swarms.tokenizers.BaseTokenizer.count_tokens",
        lambda x, y: 50,
    )
    assert base_tokenizer.count_tokens_left("some text") == 50


def test_count_tokens_left_with_zero_diff(
    base_tokenizer, monkeypatch
):
    monkeypatch.setattr(
        "swarms.tokenizers.BaseTokenizer.count_tokens",
        lambda x, y: 100,
    )
    assert base_tokenizer.count_tokens_left("some text") == 0


# 4. Add tests for count_tokens. This method is an abstract one, so testing it
# will be dependent on the actual implementation in the subclass. Here is just
# a general idea how to test it (we assume that test_count_tokens is implemented in some subclass).
def test_count_tokens(subclass_tokenizer_instance):
    assert subclass_tokenizer_instance.count_tokens("some text") == 6
