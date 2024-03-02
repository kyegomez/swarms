# HuggingFaceTokenizer

import os
from unittest.mock import patch

import pytest

from swarms.tokenizers.r_tokenizers import HuggingFaceTokenizer


# Test class setup
@pytest.fixture
def hftokenizer():
    dir_path = os.path.join(os.getcwd(), "modeldir")
    tokenizer = HuggingFaceTokenizer(dir_path)
    return tokenizer


# testing __init__
@patch("os.path")
@patch("swarms.tokenizers.get_logger")
def test___init__(mock_get_logger, mock_path, hftokenizer):
    mock_path.exists.return_value = False
    mock_path.join.return_value = "dummy_path"
    mock_get_logger.return_value = "dummy_logger"
    assert hftokenizer.model_dir == "dummy_path"
    assert hftokenizer.logger == "dummy_logger"
    assert hftokenizer._maybe_decode_bytes is False
    assert hftokenizer._prefix_space_tokens is None


# testing vocab_size property
def test_vocab_size(hftokenizer):
    assert hftokenizer.vocab_size == 30522


# testing bos_token_id property
def test_bos_token_id(hftokenizer):
    assert hftokenizer.bos_token_id == 101


# testing eos_token_id property
def test_eos_token_id(hftokenizer):
    assert hftokenizer.eos_token_id == 102


# testing prefix_space_tokens property
def test_prefix_space_tokens(hftokenizer):
    assert len(hftokenizer.prefix_space_tokens) > 0


# testing _maybe_add_prefix_space method
def test__maybe_add_prefix_space(hftokenizer):
    assert (
        hftokenizer._maybe_add_prefix_space(
            [101, 2003, 2010, 2050, 2001, 2339], " is why"
        )
        == " is why"
    )
    assert (
        hftokenizer._maybe_add_prefix_space(
            [2003, 2010, 2050, 2001, 2339], "is why"
        )
        == " is why"
    )


# continuing tests for other methods...
