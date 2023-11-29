import logging
import os
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)

from swarms.models.gpt4v import GPT4Vision, GPT4VisionResponse

load_dotenv

api_key = os.getenv("OPENAI_API_KEY")


# Mock the OpenAI client
@pytest.fixture
def mock_openai_client():
    return Mock()


@pytest.fixture
def gpt4vision(mock_openai_client):
    return GPT4Vision(client=mock_openai_client)


def test_gpt4vision_default_values():
    # Arrange and Act
    gpt4vision = GPT4Vision()

    # Assert
    assert gpt4vision.max_retries == 3
    assert gpt4vision.model == "gpt-4-vision-preview"
    assert gpt4vision.backoff_factor == 2.0
    assert gpt4vision.timeout_seconds == 10
    assert gpt4vision.api_key is None
    assert gpt4vision.quality == "low"
    assert gpt4vision.max_tokens == 200


def test_gpt4vision_api_key_from_env_variable():
    # Arrange
    api_key = os.environ["OPENAI_API_KEY"]

    # Act
    gpt4vision = GPT4Vision()

    # Assert
    assert gpt4vision.api_key == api_key


def test_gpt4vision_set_api_key():
    # Arrange
    gpt4vision = GPT4Vision(api_key=api_key)

    # Assert
    assert gpt4vision.api_key == api_key


def test_gpt4vision_invalid_max_retries():
    # Arrange and Act
    with pytest.raises(ValueError):
        GPT4Vision(max_retries=-1)


def test_gpt4vision_invalid_backoff_factor():
    # Arrange and Act
    with pytest.raises(ValueError):
        GPT4Vision(backoff_factor=-1)


def test_gpt4vision_invalid_timeout_seconds():
    # Arrange and Act
    with pytest.raises(ValueError):
        GPT4Vision(timeout_seconds=-1)


def test_gpt4vision_invalid_max_tokens():
    # Arrange and Act
    with pytest.raises(ValueError):
        GPT4Vision(max_tokens=-1)


def test_gpt4vision_logger_initialized():
    # Arrange
    gpt4vision = GPT4Vision()

    # Assert
    assert isinstance(gpt4vision.logger, logging.Logger)


def test_gpt4vision_process_img_nonexistent_file():
    # Arrange
    gpt4vision = GPT4Vision()
    img_path = "nonexistent_image.jpg"

    # Act and Assert
    with pytest.raises(FileNotFoundError):
        gpt4vision.process_img(img_path)


def test_gpt4vision_call_single_task_single_image_no_openai_client(
    gpt4vision,
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    task = "Describe this image."

    # Act and Assert
    with pytest.raises(AttributeError):
        gpt4vision(img_url, [task])


def test_gpt4vision_call_single_task_single_image_empty_response(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    task = "Describe this image."

    mock_openai_client.chat.completions.create.return_value.choices = (
        []
    )

    # Act
    response = gpt4vision(img_url, [task])

    # Assert
    assert response.answer == ""
    mock_openai_client.chat.completions.create.assert_called_once()


def test_gpt4vision_call_multiple_tasks_single_image_empty_responses(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    tasks = ["Describe this image.", "What's in this picture?"]

    mock_openai_client.chat.completions.create.return_value.choices = (
        []
    )

    # Act
    responses = gpt4vision(img_url, tasks)

    # Assert
    assert all(response.answer == "" for response in responses)
    assert (
        mock_openai_client.chat.completions.create.call_count == 1
    )  # Should be called only once


def test_gpt4vision_call_single_task_single_image_timeout(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    task = "Describe this image."

    mock_openai_client.chat.completions.create.side_effect = Timeout(
        "Request timed out"
    )

    # Act and Assert
    with pytest.raises(Timeout):
        gpt4vision(img_url, [task])


def test_gpt4vision_call_retry_with_success_after_timeout(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    task = "Describe this image."

    # Simulate success after a timeout and retry
    mock_openai_client.chat.completions.create.side_effect = [
        Timeout("Request timed out"),
        {
            "choices": [
                {
                    "message": {
                        "content": {
                            "text": "A description of the image."
                        }
                    }
                }
            ],
        },
    ]

    # Act
    response = gpt4vision(img_url, [task])

    # Assert
    assert response.answer == "A description of the image."
    assert (
        mock_openai_client.chat.completions.create.call_count == 2
    )  # Should be called twice


def test_gpt4vision_process_img():
    # Arrange
    img_path = "test_image.jpg"
    gpt4vision = GPT4Vision()

    # Act
    img_data = gpt4vision.process_img(img_path)

    # Assert
    assert img_data.startswith("/9j/")  # Base64-encoded image data


def test_gpt4vision_call_single_task_single_image(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    task = "Describe this image."

    expected_response = GPT4VisionResponse(
        answer="A description of the image."
    )

    mock_openai_client.chat.completions.create.return_value.choices[
        0
    ].text = expected_response.answer

    # Act
    response = gpt4vision(img_url, [task])

    # Assert
    assert response == expected_response
    mock_openai_client.chat.completions.create.assert_called_once()


def test_gpt4vision_call_single_task_multiple_images(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_urls = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
    ]
    task = "Describe these images."

    expected_response = GPT4VisionResponse(
        answer="Descriptions of the images."
    )

    mock_openai_client.chat.completions.create.return_value.choices[
        0
    ].text = expected_response.answer

    # Act
    response = gpt4vision(img_urls, [task])

    # Assert
    assert response == expected_response
    mock_openai_client.chat.completions.create.assert_called_once()


def test_gpt4vision_call_multiple_tasks_single_image(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    tasks = ["Describe this image.", "What's in this picture?"]

    expected_responses = [
        GPT4VisionResponse(answer="A description of the image."),
        GPT4VisionResponse(answer="It contains various objects."),
    ]

    def create_mock_response(response):
        return {
            "choices": [
                {"message": {"content": {"text": response.answer}}}
            ]
        }

    mock_openai_client.chat.completions.create.side_effect = [
        create_mock_response(response)
        for response in expected_responses
    ]

    # Act
    responses = gpt4vision(img_url, tasks)

    # Assert
    assert responses == expected_responses
    assert (
        mock_openai_client.chat.completions.create.call_count == 1
    )  # Should be called only once

    def test_gpt4vision_call_multiple_tasks_single_image(
        gpt4vision, mock_openai_client
    ):
        # Arrange
        img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        tasks = ["Describe this image.", "What's in this picture?"]

        expected_responses = [
            GPT4VisionResponse(answer="A description of the image."),
            GPT4VisionResponse(answer="It contains various objects."),
        ]

        mock_openai_client.chat.completions.create.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": {
                                "text": expected_responses[i].answer
                            }
                        }
                    }
                ]
            }
            for i in range(len(expected_responses))
        ]

        # Act
        responses = gpt4vision(img_url, tasks)

        # Assert
        assert responses == expected_responses
        assert (
            mock_openai_client.chat.completions.create.call_count == 1
        )  # Should be called only once


def test_gpt4vision_call_multiple_tasks_multiple_images(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_urls = [
        "https://images.unsplash.com/photo-1694734479857-626882b6db37?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1694734479898-6ac4633158ac?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    ]
    tasks = ["Describe these images.", "What's in these pictures?"]

    expected_responses = [
        GPT4VisionResponse(answer="Descriptions of the images."),
        GPT4VisionResponse(answer="They contain various objects."),
    ]

    mock_openai_client.chat.completions.create.side_effect = [
        {
            "choices": [
                {"message": {"content": {"text": response.answer}}}
            ]
        }
        for response in expected_responses
    ]

    # Act
    responses = gpt4vision(img_urls, tasks)

    # Assert
    assert responses == expected_responses
    assert (
        mock_openai_client.chat.completions.create.call_count == 1
    )  # Should be called only once


def test_gpt4vision_call_http_error(gpt4vision, mock_openai_client):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    task = "Describe this image."

    mock_openai_client.chat.completions.create.side_effect = (
        HTTPError("HTTP Error")
    )

    # Act and Assert
    with pytest.raises(HTTPError):
        gpt4vision(img_url, [task])


def test_gpt4vision_call_request_error(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    task = "Describe this image."

    mock_openai_client.chat.completions.create.side_effect = (
        RequestException("Request Error")
    )

    # Act and Assert
    with pytest.raises(RequestException):
        gpt4vision(img_url, [task])


def test_gpt4vision_call_connection_error(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    task = "Describe this image."

    mock_openai_client.chat.completions.create.side_effect = (
        ConnectionError("Connection Error")
    )

    # Act and Assert
    with pytest.raises(ConnectionError):
        gpt4vision(img_url, [task])


def test_gpt4vision_call_retry_with_success(
    gpt4vision, mock_openai_client
):
    # Arrange
    img_url = "https://images.unsplash.com/photo-1694734479942-8cc7f4660578?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    task = "Describe this image."

    # Simulate success after a retry
    mock_openai_client.chat.completions.create.side_effect = [
        RequestException("Temporary error"),
        {
            "choices": [{"text": "A description of the image."}]
        },  # fixed dictionary syntax
    ]

    # Act
    response = gpt4vision(img_url, [task])

    # Assert
    assert response.answer == "A description of the image."
    assert (
        mock_openai_client.chat.completions.create.call_count == 2
    )  # Should be called twice
