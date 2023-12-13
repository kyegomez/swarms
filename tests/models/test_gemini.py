import pytest
from unittest.mock import patch, Mock
from swarms.models.gemini import Gemini


# Define test fixtures
@pytest.fixture
def mock_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mocked-api-key")


@pytest.fixture
def mock_genai_model():
    return Mock()


# Test initialization of Gemini
def test_gemini_init_defaults(mock_gemini_api_key, mock_genai_model):
    model = Gemini()
    assert model.model_name == "gemini-pro"
    assert model.gemini_api_key == "mocked-api-key"
    assert model.model is mock_genai_model


def test_gemini_init_custom_params(
    mock_gemini_api_key, mock_genai_model
):
    model = Gemini(
        model_name="custom-model", gemini_api_key="custom-api-key"
    )
    assert model.model_name == "custom-model"
    assert model.gemini_api_key == "custom-api-key"
    assert model.model is mock_genai_model


# Test Gemini run method
@patch("swarms.models.gemini.Gemini.process_img")
@patch("swarms.models.gemini.genai.GenerativeModel.generate_content")
def test_gemini_run_with_img(
    mock_generate_content,
    mock_process_img,
    mock_gemini_api_key,
    mock_genai_model,
):
    model = Gemini()
    task = "A cat"
    img = "cat.png"
    response_mock = Mock(text="Generated response")
    mock_generate_content.return_value = response_mock
    mock_process_img.return_value = "Processed image"

    response = model.run(task=task, img=img)

    assert response == "Generated response"
    mock_generate_content.assert_called_with(
        content=[task, "Processed image"]
    )
    mock_process_img.assert_called_with(img=img)


@patch("swarms.models.gemini.genai.GenerativeModel.generate_content")
def test_gemini_run_without_img(
    mock_generate_content, mock_gemini_api_key, mock_genai_model
):
    model = Gemini()
    task = "A cat"
    response_mock = Mock(text="Generated response")
    mock_generate_content.return_value = response_mock

    response = model.run(task=task)

    assert response == "Generated response"
    mock_generate_content.assert_called_with(task=task)


@patch("swarms.models.gemini.genai.GenerativeModel.generate_content")
def test_gemini_run_exception(
    mock_generate_content, mock_gemini_api_key, mock_genai_model
):
    model = Gemini()
    task = "A cat"
    mock_generate_content.side_effect = Exception("Test exception")

    response = model.run(task=task)

    assert response is None


# Test Gemini process_img method
def test_gemini_process_img(mock_gemini_api_key, mock_genai_model):
    model = Gemini(gemini_api_key="custom-api-key")
    img = "cat.png"
    img_data = b"Mocked image data"

    with patch("builtins.open", create=True) as open_mock:
        open_mock.return_value.__enter__.return_value.read.return_value = (
            img_data
        )

        processed_img = model.process_img(img)

    assert processed_img == [
        {"mime_type": "image/png", "data": img_data}
    ]
    open_mock.assert_called_with(img, "rb")


# Test Gemini initialization with missing API key
def test_gemini_init_missing_api_key():
    with pytest.raises(
        ValueError, match="Please provide a Gemini API key"
    ):
        model = Gemini(gemini_api_key=None)


# Test Gemini initialization with missing model name
def test_gemini_init_missing_model_name():
    with pytest.raises(
        ValueError, match="Please provide a model name"
    ):
        model = Gemini(model_name=None)


# Test Gemini run method with empty task
def test_gemini_run_empty_task(mock_gemini_api_key, mock_genai_model):
    model = Gemini()
    task = ""
    response = model.run(task=task)
    assert response is None


# Test Gemini run method with empty image
def test_gemini_run_empty_img(mock_gemini_api_key, mock_genai_model):
    model = Gemini()
    task = "A cat"
    img = ""
    response = model.run(task=task, img=img)
    assert response is None


# Test Gemini process_img method with missing image
def test_gemini_process_img_missing_image(
    mock_gemini_api_key, mock_genai_model
):
    model = Gemini()
    img = None
    with pytest.raises(
        ValueError, match="Please provide an image to process"
    ):
        model.process_img(img=img)


# Test Gemini process_img method with missing image type
def test_gemini_process_img_missing_image_type(
    mock_gemini_api_key, mock_genai_model
):
    model = Gemini()
    img = "cat.png"
    with pytest.raises(
        ValueError, match="Please provide the image type"
    ):
        model.process_img(img=img, type=None)


# Test Gemini process_img method with missing Gemini API key
def test_gemini_process_img_missing_api_key(mock_genai_model):
    model = Gemini(gemini_api_key=None)
    img = "cat.png"
    with pytest.raises(
        ValueError, match="Please provide a Gemini API key"
    ):
        model.process_img(img=img, type="image/png")


# Test Gemini run method with mocked image processing
@patch("swarms.models.gemini.genai.GenerativeModel.generate_content")
@patch("swarms.models.gemini.Gemini.process_img")
def test_gemini_run_mock_img_processing(
    mock_process_img,
    mock_generate_content,
    mock_gemini_api_key,
    mock_genai_model,
):
    model = Gemini()
    task = "A cat"
    img = "cat.png"
    response_mock = Mock(text="Generated response")
    mock_generate_content.return_value = response_mock
    mock_process_img.return_value = "Processed image"

    response = model.run(task=task, img=img)

    assert response == "Generated response"
    mock_generate_content.assert_called_with(
        content=[task, "Processed image"]
    )
    mock_process_img.assert_called_with(img=img)


# Test Gemini run method with mocked image processing and exception
@patch("swarms.models.gemini.Gemini.process_img")
@patch("swarms.models.gemini.genai.GenerativeModel.generate_content")
def test_gemini_run_mock_img_processing_exception(
    mock_generate_content,
    mock_process_img,
    mock_gemini_api_key,
    mock_genai_model,
):
    model = Gemini()
    task = "A cat"
    img = "cat.png"
    mock_process_img.side_effect = Exception("Test exception")

    response = model.run(task=task, img=img)

    assert response is None
    mock_generate_content.assert_not_called()
    mock_process_img.assert_called_with(img=img)
