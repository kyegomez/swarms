import pytest
from swarms.chunkers.base import (
    BaseChunker,
    TextArtifact,
    ChunkSeparator,
    OpenAiTokenizer,
)  # adjust the import paths accordingly


# 1. Test Initialization
def test_chunker_initialization():
    chunker = BaseChunker()
    assert isinstance(chunker, BaseChunker)
    assert chunker.max_tokens == chunker.tokenizer.max_tokens


def test_default_separators():
    chunker = BaseChunker()
    assert chunker.separators == BaseChunker.DEFAULT_SEPARATORS


def test_default_tokenizer():
    chunker = BaseChunker()
    assert isinstance(chunker.tokenizer, OpenAiTokenizer)


# 2. Test Basic Chunking
@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("This is a test.", [TextArtifact("This is a test.")]),
        ("Hello World!", [TextArtifact("Hello World!")]),
        # Add more simple cases
    ],
)
def test_basic_chunk(input_text, expected_output):
    chunker = BaseChunker()
    result = chunker.chunk(input_text)
    assert result == expected_output


# 3. Test Chunking with Different Separators
def test_custom_separators():
    custom_separator = ChunkSeparator(";")
    chunker = BaseChunker(separators=[custom_separator])
    input_text = "Hello;World!"
    expected_output = [TextArtifact("Hello;"), TextArtifact("World!")]
    result = chunker.chunk(input_text)
    assert result == expected_output


# 4. Test Recursive Chunking
def test_recursive_chunking():
    chunker = BaseChunker(max_tokens=5)
    input_text = "This is a more complex text."
    expected_output = [
        TextArtifact("This"),
        TextArtifact("is a"),
        TextArtifact("more"),
        TextArtifact("complex"),
        TextArtifact("text."),
    ]
    result = chunker.chunk(input_text)
    assert result == expected_output


# 5. Test Edge Cases and Special Scenarios
def test_empty_text():
    chunker = BaseChunker()
    result = chunker.chunk("")
    assert result == []


def test_whitespace_text():
    chunker = BaseChunker()
    result = chunker.chunk("     ")
    assert result == [TextArtifact("     ")]


def test_single_word():
    chunker = BaseChunker()
    result = chunker.chunk("Hello")
    assert result == [TextArtifact("Hello")]
