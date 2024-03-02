# CohereTokenizer

from unittest.mock import MagicMock

import pytest

from swarms.tokenizers.cohere_tokenizer import CohereTokenizer


@pytest.fixture
def cohere_tokenizer():
    mock_client = MagicMock()
    mock_client.tokenize.return_value.tokens = [
        "token1",
        "token2",
        "token3",
    ]
    return CohereTokenizer(model="<model-name>", client=mock_client)


def test_count_tokens_with_string(cohere_tokenizer):
    tokens_count = cohere_tokenizer.count_tokens("valid string")
    assert tokens_count == 3


def test_count_tokens_with_non_string(cohere_tokenizer):
    with pytest.raises(ValueError):
        cohere_tokenizer.count_tokens(["invalid", "input"])


def test_count_tokens_with_different_length(cohere_tokenizer):
    cohere_tokenizer.client.tokenize.return_value.tokens = [
        "token1",
        "token2",
    ]
    tokens_count = cohere_tokenizer.count_tokens("valid string")
    assert tokens_count == 2
