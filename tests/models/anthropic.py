import os
import pytest
from unittest.mock import Mock, patch
from swarms.models.anthropic import Anthropic


@pytest.fixture
def mock_anthropic_env():
    os.environ["ANTHROPIC_API_URL"] = "https://test.anthropic.com"
    os.environ["ANTHROPIC_API_KEY"] = "test_api_key"
    yield
    del os.environ["ANTHROPIC_API_URL"]
    del os.environ["ANTHROPIC_API_KEY"]


@pytest.fixture
def mock_requests_post():
    with patch("requests.post") as mock_post:
        yield mock_post


@pytest.fixture
def anthropic_instance():
    return Anthropic(model="test-model")


def test_anthropic_init_default_values(anthropic_instance):
    assert anthropic_instance.model == "test-model"
    assert anthropic_instance.max_tokens_to_sample == 256
    assert anthropic_instance.temperature is None
    assert anthropic_instance.top_k is None
    assert anthropic_instance.top_p is None
    assert anthropic_instance.streaming is False
    assert anthropic_instance.default_request_timeout == 600
    assert anthropic_instance.anthropic_api_url == "https://test.anthropic.com"
    assert anthropic_instance.anthropic_api_key == "test_api_key"


def test_anthropic_init_custom_values():
    anthropic_instance = Anthropic(
        model="custom-model",
        max_tokens_to_sample=128,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        streaming=True,
        default_request_timeout=300,
    )
    assert anthropic_instance.model == "custom-model"
    assert anthropic_instance.max_tokens_to_sample == 128
    assert anthropic_instance.temperature == 0.8
    assert anthropic_instance.top_k == 5
    assert anthropic_instance.top_p == 0.9
    assert anthropic_instance.streaming is True
    assert anthropic_instance.default_request_timeout == 300


def test_anthropic_default_params(anthropic_instance):
    default_params = anthropic_instance._default_params()
    assert default_params == {
        "max_tokens_to_sample": 256,
        "model": "test-model",
    }


def test_anthropic_run(mock_anthropic_env, mock_requests_post, anthropic_instance):
    mock_response = Mock()
    mock_response.json.return_value = {"completion": "Generated text"}
    mock_requests_post.return_value = mock_response

    task = "Generate text"
    stop = ["stop1", "stop2"]

    completion = anthropic_instance.run(task, stop)

    assert completion == "Generated text"
    mock_requests_post.assert_called_once_with(
        "https://test.anthropic.com/completions",
        headers={"Authorization": "Bearer test_api_key"},
        json={
            "prompt": task,
            "stop_sequences": stop,
            "max_tokens_to_sample": 256,
            "model": "test-model",
        },
        timeout=600,
    )


def test_anthropic_call(mock_anthropic_env, mock_requests_post, anthropic_instance):
    mock_response = Mock()
    mock_response.json.return_value = {"completion": "Generated text"}
    mock_requests_post.return_value = mock_response

    task = "Generate text"
    stop = ["stop1", "stop2"]

    completion = anthropic_instance(task, stop)

    assert completion == "Generated text"
    mock_requests_post.assert_called_once_with(
        "https://test.anthropic.com/completions",
        headers={"Authorization": "Bearer test_api_key"},
        json={
            "prompt": task,
            "stop_sequences": stop,
            "max_tokens_to_sample": 256,
            "model": "test-model",
        },
        timeout=600,
    )


def test_anthropic_exception_handling(
    mock_anthropic_env, mock_requests_post, anthropic_instance
):
    mock_response = Mock()
    mock_response.json.return_value = {"error": "An error occurred"}
    mock_requests_post.return_value = mock_response

    task = "Generate text"
    stop = ["stop1", "stop2"]

    with pytest.raises(Exception) as excinfo:
        anthropic_instance(task, stop)

    assert "An error occurred" in str(excinfo.value)
