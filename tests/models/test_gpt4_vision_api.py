import pytest
from unittest.mock import mock_open, patch, Mock
from requests.exceptions import RequestException
from swarms.models.gpt4_vision_api import GPT4VisionAPI
import os
from dotenv import load_dotenv

load_dotenv()


custom_api_key = os.environ.get("OPENAI_API_KEY")
img = "images/swarms.jpeg"


@pytest.fixture
def vision_api():
    return GPT4VisionAPI(openai_api_key="test_api_key")


def test_init(vision_api):
    assert vision_api.openai_api_key == "test_api_key"


def test_encode_image(vision_api):
    with patch(
        "builtins.open", mock_open(read_data=b"test_image_data"), create=True
    ):
        encoded_image = vision_api.encode_image("test_image.jpg")
        assert encoded_image == "dGVzdF9pbWFnZV9kYXRh"


def test_run_success(vision_api):
    expected_response = {"choices": [{"text": "This is the model's response."}]}
    with patch(
        "requests.post", return_value=Mock(json=lambda: expected_response)
    ) as mock_post:
        result = vision_api.run("What is this?", "test_image.jpg")
        mock_post.assert_called_once()
        assert result == "This is the model's response."


def test_run_request_error(vision_api):
    with patch(
        "requests.post", side_effect=RequestException("Request Error")
    ) as mock_post:
        with pytest.raises(RequestException):
            vision_api.run("What is this?", "test_image.jpg")


def test_run_response_error(vision_api):
    expected_response = {"error": "Model Error"}
    with patch(
        "requests.post", return_value=Mock(json=lambda: expected_response)
    ) as mock_post:
        with pytest.raises(RuntimeError):
            vision_api.run("What is this?", "test_image.jpg")


def test_call(vision_api):
    expected_response = {"choices": [{"text": "This is the model's response."}]}
    with patch(
        "requests.post", return_value=Mock(json=lambda: expected_response)
    ) as mock_post:
        result = vision_api("What is this?", "test_image.jpg")
        mock_post.assert_called_once()
        assert result == "This is the model's response."


@pytest.fixture
def gpt_api():
    return GPT4VisionAPI()


def test_initialization_with_default_key():
    api = GPT4VisionAPI()
    assert api.openai_api_key == custom_api_key


def test_initialization_with_custom_key():
    custom_key = custom_api_key
    api = GPT4VisionAPI(openai_api_key=custom_key)
    assert api.openai_api_key == custom_key


def test_run_successful_response(gpt_api):
    task = "What is in the image?"
    img_url = img
    response_json = {"choices": [{"text": "Answer from GPT-4 Vision"}]}
    mock_response = Mock()
    mock_response.json.return_value = response_json
    with patch("requests.post", return_value=mock_response) as mock_post:
        result = gpt_api.run(task, img_url)
        mock_post.assert_called_once()
    assert result == response_json["choices"][0]["text"]


def test_run_with_exception(gpt_api):
    task = "What is in the image?"
    img_url = img
    with patch("requests.post", side_effect=Exception("Test Exception")):
        with pytest.raises(Exception):
            gpt_api.run(task, img_url)


def test_call_method_successful_response(gpt_api):
    task = "What is in the image?"
    img_url = img
    response_json = {"choices": [{"text": "Answer from GPT-4 Vision"}]}
    mock_response = Mock()
    mock_response.json.return_value = response_json
    with patch("requests.post", return_value=mock_response) as mock_post:
        result = gpt_api(task, img_url)
        mock_post.assert_called_once()
    assert result == response_json


def test_call_method_with_exception(gpt_api):
    task = "What is in the image?"
    img_url = img
    with patch("requests.post", side_effect=Exception("Test Exception")):
        with pytest.raises(Exception):
            gpt_api(task, img_url)
