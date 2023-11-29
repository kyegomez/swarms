# test_embeddings.py

import pytest
import openai
from unittest.mock import patch
from swarms.models.simple_ada import (
    get_ada_embeddings,
)  # Adjust this import path to your project structure
from os import getenv
from dotenv import load_dotenv

load_dotenv()


# Fixture for test texts
@pytest.fixture
def test_texts():
    return [
        "Hello World",
        "This is a test string with newline\ncharacters",
        "A quick brown fox jumps over the lazy dog",
    ]


# Basic Test
def test_get_ada_embeddings_basic(test_texts):
    with patch("openai.resources.Embeddings.create") as mock_create:
        # Mocking the OpenAI API call
        mock_create.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        for text in test_texts:
            embedding = get_ada_embeddings(text)
            assert embedding == [
                0.1,
                0.2,
                0.3,
            ], "Embedding does not match expected output"
            mock_create.assert_called_with(
                input=[text.replace("\n", " ")],
                model="text-embedding-ada-002",
            )


# Parameterized Test
@pytest.mark.parametrize(
    "text, model, expected_call_model",
    [
        (
            "Hello World",
            "text-embedding-ada-002",
            "text-embedding-ada-002",
        ),
        (
            "Hello World",
            "text-embedding-ada-001",
            "text-embedding-ada-001",
        ),
    ],
)
def test_get_ada_embeddings_models(text, model, expected_call_model):
    with patch("openai.resources.Embeddings.create") as mock_create:
        mock_create.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        _ = get_ada_embeddings(text, model=model)
        mock_create.assert_called_with(
            input=[text], model=expected_call_model
        )


# Exception Test
def test_get_ada_embeddings_exception():
    with patch("openai.resources.Embeddings.create") as mock_create:
        mock_create.side_effect = openai.OpenAIError("Test error")
        with pytest.raises(openai.OpenAIError):
            get_ada_embeddings("Some text")


# Tests for environment variable loading
def test_env_var_loading(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "testkey123")
    with patch("openai.resources.Embeddings.create"):
        assert (
            getenv("OPENAI_API_KEY") == "testkey123"
        ), "Environment variable for API key is not set correctly"


# ... more tests to cover other aspects such as different input types, large inputs, invalid inputs, etc.
