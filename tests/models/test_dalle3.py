import os
from unittest.mock import Mock

import pytest
from openai import OpenAIError
from PIL import Image
from termcolor import colored

from swarms.models.dalle3 import Dalle3


# Mocking the OpenAI client to avoid making actual API calls during testing
@pytest.fixture
def mock_openai_client():
    return Mock()


@pytest.fixture
def dalle3(mock_openai_client):
    return Dalle3(client=mock_openai_client)


def test_dalle3_call_success(dalle3, mock_openai_client):
    # Arrange
    task = "A painting of a dog"
    expected_img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
    mock_openai_client.images.generate.return_value = Mock(
        data=[Mock(url=expected_img_url)]
    )

    # Act
    img_url = dalle3(task)

    # Assert
    assert img_url == expected_img_url
    mock_openai_client.images.generate.assert_called_once_with(
        prompt=task, n=4
    )


def test_dalle3_call_failure(dalle3, mock_openai_client, capsys):
    # Arrange
    task = "Invalid task"
    expected_error_message = "Error running Dalle3: API Error"

    # Mocking OpenAIError
    mock_openai_client.images.generate.side_effect = OpenAIError(
        expected_error_message,
        http_status=500,
        error="Internal Server Error",
    )

    # Act and assert
    with pytest.raises(OpenAIError) as excinfo:
        dalle3(task)

    assert str(excinfo.value) == expected_error_message
    mock_openai_client.images.generate.assert_called_once_with(
        prompt=task, n=4
    )

    # Ensure the error message is printed in red
    captured = capsys.readouterr()
    assert colored(expected_error_message, "red") in captured.out


def test_dalle3_create_variations_success(dalle3, mock_openai_client):
    # Arrange
    img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
    expected_variation_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_02ABCDE.png"
    mock_openai_client.images.create_variation.return_value = Mock(
        data=[Mock(url=expected_variation_url)]
    )

    # Act
    variation_img_url = dalle3.create_variations(img_url)

    # Assert
    assert variation_img_url == expected_variation_url
    mock_openai_client.images.create_variation.assert_called_once()
    _, kwargs = mock_openai_client.images.create_variation.call_args
    assert kwargs["img"] is not None
    assert kwargs["n"] == 4
    assert kwargs["size"] == "1024x1024"


def test_dalle3_create_variations_failure(
    dalle3, mock_openai_client, capsys
):
    # Arrange
    img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
    expected_error_message = "Error running Dalle3: API Error"

    # Mocking OpenAIError
    mock_openai_client.images.create_variation.side_effect = (
        OpenAIError(
            expected_error_message,
            http_status=500,
            error="Internal Server Error",
        )
    )

    # Act and assert
    with pytest.raises(OpenAIError) as excinfo:
        dalle3.create_variations(img_url)

    assert str(excinfo.value) == expected_error_message
    mock_openai_client.images.create_variation.assert_called_once()

    # Ensure the error message is printed in red
    captured = capsys.readouterr()
    assert colored(expected_error_message, "red") in captured.out


def test_dalle3_read_img():
    # Arrange
    img_path = "test_image.png"
    img = Image.new("RGB", (512, 512))

    # Save the image temporarily
    img.save(img_path)

    # Act
    dalle3 = Dalle3()
    img_loaded = dalle3.read_img(img_path)

    # Assert
    assert isinstance(img_loaded, Image.Image)

    # Clean up
    os.remove(img_path)


def test_dalle3_set_width_height():
    # Arrange
    img = Image.new("RGB", (512, 512))
    width = 256
    height = 256

    # Act
    dalle3 = Dalle3()
    img_resized = dalle3.set_width_height(img, width, height)

    # Assert
    assert img_resized.size == (width, height)


def test_dalle3_convert_to_bytesio():
    # Arrange
    img = Image.new("RGB", (512, 512))
    expected_format = "PNG"

    # Act
    dalle3 = Dalle3()
    img_bytes = dalle3.convert_to_bytesio(img, format=expected_format)

    # Assert
    assert isinstance(img_bytes, bytes)
    assert img_bytes.startswith(b"\x89PNG")


def test_dalle3_call_multiple_times(dalle3, mock_openai_client):
    # Arrange
    task = "A painting of a dog"
    expected_img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
    mock_openai_client.images.generate.return_value = Mock(
        data=[Mock(url=expected_img_url)]
    )

    # Act
    img_url1 = dalle3(task)
    img_url2 = dalle3(task)

    # Assert
    assert img_url1 == expected_img_url
    assert img_url2 == expected_img_url
    assert mock_openai_client.images.generate.call_count == 2


def test_dalle3_call_with_large_input(dalle3, mock_openai_client):
    # Arrange
    task = "A" * 2048  # Input longer than API's limit
    expected_error_message = "Error running Dalle3: API Error"
    mock_openai_client.images.generate.side_effect = OpenAIError(
        expected_error_message,
        http_status=500,
        error="Internal Server Error",
    )

    # Act and assert
    with pytest.raises(OpenAIError) as excinfo:
        dalle3(task)

    assert str(excinfo.value) == expected_error_message


def test_dalle3_create_variations_with_invalid_image_url(
    dalle3, mock_openai_client
):
    # Arrange
    img_url = "https://invalid-image-url.com"
    expected_error_message = "Error running Dalle3: Invalid image URL"

    # Act and assert
    with pytest.raises(ValueError) as excinfo:
        dalle3.create_variations(img_url)

    assert str(excinfo.value) == expected_error_message


def test_dalle3_set_width_height_invalid_dimensions(dalle3):
    # Arrange
    img = dalle3.read_img("test_image.png")
    width = 0
    height = -1

    # Act and assert
    with pytest.raises(ValueError):
        dalle3.set_width_height(img, width, height)


def test_dalle3_convert_to_bytesio_invalid_format(dalle3):
    # Arrange
    img = dalle3.read_img("test_image.png")
    invalid_format = "invalid_format"

    # Act and assert
    with pytest.raises(ValueError):
        dalle3.convert_to_bytesio(img, format=invalid_format)


def test_dalle3_call_with_retry(dalle3, mock_openai_client):
    # Arrange
    task = "A painting of a dog"
    expected_img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"

    # Simulate a retry scenario
    mock_openai_client.images.generate.side_effect = [
        OpenAIError(
            "Temporary error",
            http_status=500,
            error="Internal Server Error",
        ),
        Mock(data=[Mock(url=expected_img_url)]),
    ]

    # Act
    img_url = dalle3(task)

    # Assert
    assert img_url == expected_img_url
    assert mock_openai_client.images.generate.call_count == 2


def test_dalle3_create_variations_with_retry(
    dalle3, mock_openai_client
):
    # Arrange
    img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
    expected_variation_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_02ABCDE.png"

    # Simulate a retry scenario
    mock_openai_client.images.create_variation.side_effect = [
        OpenAIError(
            "Temporary error",
            http_status=500,
            error="Internal Server Error",
        ),
        Mock(data=[Mock(url=expected_variation_url)]),
    ]

    # Act
    variation_img_url = dalle3.create_variations(img_url)

    # Assert
    assert variation_img_url == expected_variation_url
    assert mock_openai_client.images.create_variation.call_count == 2


def test_dalle3_call_exception_logging(
    dalle3, mock_openai_client, capsys
):
    # Arrange
    task = "A painting of a dog"
    expected_error_message = "Error running Dalle3: API Error"

    # Mocking OpenAIError
    mock_openai_client.images.generate.side_effect = OpenAIError(
        expected_error_message,
        http_status=500,
        error="Internal Server Error",
    )

    # Act
    with pytest.raises(OpenAIError):
        dalle3(task)

    # Assert that the error message is logged
    captured = capsys.readouterr()
    assert expected_error_message in captured.err


def test_dalle3_create_variations_exception_logging(
    dalle3, mock_openai_client, capsys
):
    # Arrange
    img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
    expected_error_message = "Error running Dalle3: API Error"

    # Mocking OpenAIError
    mock_openai_client.images.create_variation.side_effect = (
        OpenAIError(
            expected_error_message,
            http_status=500,
            error="Internal Server Error",
        )
    )

    # Act
    with pytest.raises(OpenAIError):
        dalle3.create_variations(img_url)

    # Assert that the error message is logged
    captured = capsys.readouterr()
    assert expected_error_message in captured.err


def test_dalle3_read_img_invalid_path(dalle3):
    # Arrange
    invalid_img_path = "invalid_image_path.png"

    # Act and assert
    with pytest.raises(FileNotFoundError):
        dalle3.read_img(invalid_img_path)


def test_dalle3_call_no_api_key():
    # Arrange
    task = "A painting of a dog"
    dalle3 = Dalle3(api_key=None)
    expected_error_message = (
        "Error running Dalle3: API Key is missing"
    )

    # Act and assert
    with pytest.raises(ValueError) as excinfo:
        dalle3(task)

    assert str(excinfo.value) == expected_error_message


def test_dalle3_create_variations_no_api_key():
    # Arrange
    img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
    dalle3 = Dalle3(api_key=None)
    expected_error_message = (
        "Error running Dalle3: API Key is missing"
    )

    # Act and assert
    with pytest.raises(ValueError) as excinfo:
        dalle3.create_variations(img_url)

    assert str(excinfo.value) == expected_error_message


def test_dalle3_call_with_retry_max_retries_exceeded(
    dalle3, mock_openai_client
):
    # Arrange
    task = "A painting of a dog"

    # Simulate max retries exceeded
    mock_openai_client.images.generate.side_effect = OpenAIError(
        "Temporary error",
        http_status=500,
        error="Internal Server Error",
    )

    # Act and assert
    with pytest.raises(OpenAIError) as excinfo:
        dalle3(task)

    assert "Retry limit exceeded" in str(excinfo.value)


def test_dalle3_create_variations_with_retry_max_retries_exceeded(
    dalle3, mock_openai_client
):
    # Arrange
    img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"

    # Simulate max retries exceeded
    mock_openai_client.images.create_variation.side_effect = (
        OpenAIError(
            "Temporary error",
            http_status=500,
            error="Internal Server Error",
        )
    )

    # Act and assert
    with pytest.raises(OpenAIError) as excinfo:
        dalle3.create_variations(img_url)

    assert "Retry limit exceeded" in str(excinfo.value)


def test_dalle3_call_retry_with_success(dalle3, mock_openai_client):
    # Arrange
    task = "A painting of a dog"
    expected_img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"

    # Simulate success after a retry
    mock_openai_client.images.generate.side_effect = [
        OpenAIError(
            "Temporary error",
            http_status=500,
            error="Internal Server Error",
        ),
        Mock(data=[Mock(url=expected_img_url)]),
    ]

    # Act
    img_url = dalle3(task)

    # Assert
    assert img_url == expected_img_url
    assert mock_openai_client.images.generate.call_count == 2


def test_dalle3_create_variations_retry_with_success(
    dalle3, mock_openai_client
):
    # Arrange
    img_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
    expected_variation_url = "https://cdn.openai.com/dall-e/encoded/feats/feats_02ABCDE.png"

    # Simulate success after a retry
    mock_openai_client.images.create_variation.side_effect = [
        OpenAIError(
            "Temporary error",
            http_status=500,
            error="Internal Server Error",
        ),
        Mock(data=[Mock(url=expected_variation_url)]),
    ]

    # Act
    variation_img_url = dalle3.create_variations(img_url)

    # Assert
    assert variation_img_url == expected_variation_url
    assert mock_openai_client.images.create_variation.call_count == 2
