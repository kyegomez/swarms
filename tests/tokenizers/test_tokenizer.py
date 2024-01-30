# Tokenizer

from swarms.tokenizers.r_tokenizers import Tokenizer
from unittest.mock import patch


def test_initializer_existing_model_file():
    with patch("os.path.exists", return_value=True):
        with patch(
            "swarms.tokenizers.SentencePieceTokenizer"
        ) as mock_model:
            tokenizer = Tokenizer("tokenizers/my_model.model")
            mock_model.assert_called_with("tokenizers/my_model.model")
            assert tokenizer.model == mock_model.return_value


def test_initializer_model_folder():
    with patch("os.path.exists", side_effect=[False, True]):
        with patch(
            "swarms.tokenizers.HuggingFaceTokenizer"
        ) as mock_model:
            tokenizer = Tokenizer("my_model_directory")
            mock_model.assert_called_with("my_model_directory")
            assert tokenizer.model == mock_model.return_value


def test_vocab_size():
    with patch(
        "swarms.tokenizers.SentencePieceTokenizer"
    ) as mock_model:
        tokenizer = Tokenizer("tokenizers/my_model.model")
        assert (
            tokenizer.vocab_size == mock_model.return_value.vocab_size
        )


def test_bos_token_id():
    with patch(
        "swarms.tokenizers.SentencePieceTokenizer"
    ) as mock_model:
        tokenizer = Tokenizer("tokenizers/my_model.model")
        assert (
            tokenizer.bos_token_id
            == mock_model.return_value.bos_token_id
        )


def test_encode():
    with patch(
        "swarms.tokenizers.SentencePieceTokenizer"
    ) as mock_model:
        tokenizer = Tokenizer("tokenizers/my_model.model")
        assert (
            tokenizer.encode("hello")
            == mock_model.return_value.encode.return_value
        )


def test_decode():
    with patch(
        "swarms.tokenizers.SentencePieceTokenizer"
    ) as mock_model:
        tokenizer = Tokenizer("tokenizers/my_model.model")
        assert (
            tokenizer.decode([1, 2, 3])
            == mock_model.return_value.decode.return_value
        )


def test_call():
    with patch(
        "swarms.tokenizers.SentencePieceTokenizer"
    ) as mock_model:
        tokenizer = Tokenizer("tokenizers/my_model.model")
        assert (
            tokenizer("hello")
            == mock_model.return_value.__call__.return_value
        )


# More tests can be added here
