# LangchainChromaVectorMemory

from unittest.mock import MagicMock, patch

import pytest

from swarms.memory import LangchainChromaVectorMemory


# Fixtures for setting up the memory and mocks
@pytest.fixture()
def vector_memory(tmp_path):
    loc = tmp_path / "vector_memory"
    return LangchainChromaVectorMemory(loc=loc)


@pytest.fixture()
def embeddings_mock():
    with patch("swarms.memory.OpenAIEmbeddings") as mock:
        yield mock


@pytest.fixture()
def chroma_mock():
    with patch("swarms.memory.Chroma") as mock:
        yield mock


@pytest.fixture()
def qa_mock():
    with patch("swarms.memory.RetrievalQA") as mock:
        yield mock


# Example test cases
def test_initialization_default_settings(vector_memory):
    assert vector_memory.chunk_size == 1000
    assert (
        vector_memory.chunk_overlap == 100
    )  # assuming default overlap of 0.1
    assert vector_memory.loc.exists()


def test_add_entry(vector_memory, embeddings_mock):
    with patch.object(vector_memory.db, "add_texts") as add_texts_mock:
        vector_memory.add("Example text")
        add_texts_mock.assert_called()


def test_search_memory_returns_list(vector_memory):
    result = vector_memory.search_memory("example query", k=5)
    assert isinstance(result, list)


def test_ask_question_returns_string(vector_memory, qa_mock):
    result = vector_memory.query("What is the color of the sky?")
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "query,k,type,expected",
    [
        ("example query", 5, "mmr", [MagicMock()]),
        (
            "example query",
            0,
            "mmr",
            None,
        ),  # Expected none when k is 0 or negative
        (
            "example query",
            3,
            "cos",
            [MagicMock()],
        ),  # Mocked object as a placeholder
    ],
)
def test_search_memory_different_params(
    vector_memory, query, k, type, expected
):
    with patch.object(
        vector_memory.db,
        "max_marginal_relevance_search",
        return_value=expected,
    ):
        with patch.object(
            vector_memory.db,
            "similarity_search_with_score",
            return_value=expected,
        ):
            result = vector_memory.search_memory(query, k=k, type=type)
            assert len(result) == (k if k > 0 else 0)
