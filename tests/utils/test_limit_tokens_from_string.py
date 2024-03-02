import pytest

from swarms.utils import limit_tokens_from_string


def test_limit_tokens_from_string():
    sentence = (
        "This is a test sentence. It is used for testing the number"
        " of tokens."
    )
    limited = limit_tokens_from_string(sentence, limit=5)
    assert (
        len(limited.split()) <= 5
    ), "The output string has more than 5 tokens."


def test_limit_zero_tokens():
    sentence = "Expect empty result when limit is set to zero."
    limited = limit_tokens_from_string(sentence, limit=0)
    assert limited == "", "The output is not empty."


def test_negative_token_limit():
    sentence = (
        "This test will raise an exception when limit is negative."
    )
    with pytest.raises(Exception):
        limit_tokens_from_string(sentence, limit=-1)


@pytest.mark.parametrize(
    "sentence, model", [("Some sentence", "unavailable-model")]
)
def test_unknown_model(sentence, model):
    with pytest.raises(Exception):
        limit_tokens_from_string(sentence, model=model)


def test_string_token_limit_exceeded():
    sentence = (
        "This is a long sentence with more than twenty tokens which"
        " is used for testing. It checks whether the function"
        " correctly limits the tokens to a specified amount."
    )
    limited = limit_tokens_from_string(sentence, limit=20)
    assert len(limited.split()) <= 20, "The token limit is exceeded."
