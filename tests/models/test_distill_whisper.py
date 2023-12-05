import os
import tempfile
from functools import wraps
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from swarms.models.distilled_whisperx import (
    DistilWhisperModel,
    async_retry,
)


@pytest.fixture
def distil_whisper_model():
    return DistilWhisperModel()


def create_audio_file(
    data: np.ndarray, sample_rate: int, file_path: str
):
    data.tofile(file_path)
    return file_path


def test_initialization(distil_whisper_model):
    assert isinstance(distil_whisper_model, DistilWhisperModel)
    assert isinstance(distil_whisper_model.model, torch.nn.Module)
    assert isinstance(distil_whisper_model.processor, torch.nn.Module)
    assert distil_whisper_model.device in ["cpu", "cuda:0"]


def test_transcribe_audio_file(distil_whisper_model):
    test_data = np.random.rand(
        16000
    )  # Simulated audio data (1 second)
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as audio_file:
        audio_file_path = create_audio_file(
            test_data, 16000, audio_file.name
        )
        transcription = distil_whisper_model.transcribe(
            audio_file_path
        )
        os.remove(audio_file_path)

    assert isinstance(transcription, str)
    assert transcription.strip() != ""


@pytest.mark.asyncio
async def test_async_transcribe_audio_file(distil_whisper_model):
    test_data = np.random.rand(
        16000
    )  # Simulated audio data (1 second)
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as audio_file:
        audio_file_path = create_audio_file(
            test_data, 16000, audio_file.name
        )
        transcription = await distil_whisper_model.async_transcribe(
            audio_file_path
        )
        os.remove(audio_file_path)

    assert isinstance(transcription, str)
    assert transcription.strip() != ""


def test_transcribe_audio_data(distil_whisper_model):
    test_data = np.random.rand(
        16000
    )  # Simulated audio data (1 second)
    transcription = distil_whisper_model.transcribe(
        test_data.tobytes()
    )

    assert isinstance(transcription, str)
    assert transcription.strip() != ""


@pytest.mark.asyncio
async def test_async_transcribe_audio_data(distil_whisper_model):
    test_data = np.random.rand(
        16000
    )  # Simulated audio data (1 second)
    transcription = await distil_whisper_model.async_transcribe(
        test_data.tobytes()
    )

    assert isinstance(transcription, str)
    assert transcription.strip() != ""


def test_real_time_transcribe(distil_whisper_model, capsys):
    test_data = np.random.rand(
        16000 * 5
    )  # Simulated audio data (5 seconds)
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as audio_file:
        audio_file_path = create_audio_file(
            test_data, 16000, audio_file.name
        )

        distil_whisper_model.real_time_transcribe(
            audio_file_path, chunk_duration=1
        )

        os.remove(audio_file_path)

    captured = capsys.readouterr()
    assert "Starting real-time transcription..." in captured.out
    assert "Chunk" in captured.out


def test_real_time_transcribe_audio_file_not_found(
    distil_whisper_model, capsys
):
    audio_file_path = "non_existent_audio.wav"
    distil_whisper_model.real_time_transcribe(
        audio_file_path, chunk_duration=1
    )

    captured = capsys.readouterr()
    assert "The audio file was not found." in captured.out


@pytest.fixture
def mock_async_retry():
    def _mock_async_retry(
        retries=3, exceptions=(Exception,), delay=1
    ):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    with patch(
        "distil_whisper_model.async_retry", new=_mock_async_retry()
    ):
        yield


@pytest.mark.asyncio
async def test_async_retry_decorator_success():
    async def mock_async_function():
        return "Success"

    decorated_function = async_retry()(mock_async_function)
    result = await decorated_function()
    assert result == "Success"


@pytest.mark.asyncio
async def test_async_retry_decorator_failure():
    async def mock_async_function():
        raise Exception("Error")

    decorated_function = async_retry()(mock_async_function)
    with pytest.raises(Exception, match="Error"):
        await decorated_function()


@pytest.mark.asyncio
async def test_async_retry_decorator_multiple_attempts():
    async def mock_async_function():
        if mock_async_function.attempts == 0:
            mock_async_function.attempts += 1
            raise Exception("Error")
        else:
            return "Success"

    mock_async_function.attempts = 0
    decorated_function = async_retry(max_retries=2)(
        mock_async_function
    )
    result = await decorated_function()
    assert result == "Success"


def test_create_audio_file():
    test_data = np.random.rand(
        16000
    )  # Simulated audio data (1 second)
    sample_rate = 16000
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as audio_file:
        audio_file_path = create_audio_file(
            test_data, sample_rate, audio_file.name
        )

        assert os.path.exists(audio_file_path)
        os.remove(audio_file_path)


# test_distilled_whisperx.py


# Fixtures for setting up model, processor, and audio files
@pytest.fixture(scope="module")
def model_id():
    return "distil-whisper/distil-large-v2"


@pytest.fixture(scope="module")
def whisper_model(model_id):
    return DistilWhisperModel(model_id)


@pytest.fixture(scope="session")
def audio_file_path(tmp_path_factory):
    # You would create a small temporary MP3 file here for testing
    # or use a public domain MP3 file's path
    return "path/to/valid_audio.mp3"


@pytest.fixture(scope="session")
def invalid_audio_file_path():
    return "path/to/invalid_audio.mp3"


@pytest.fixture(scope="session")
def audio_dict():
    # This should represent a valid audio dictionary as expected by the model
    return {"array": torch.randn(1, 16000), "sampling_rate": 16000}


# Test initialization
def test_initialization(whisper_model):
    assert whisper_model.model is not None
    assert whisper_model.processor is not None


# Test successful transcription with file path
def test_transcribe_with_file_path(whisper_model, audio_file_path):
    transcription = whisper_model.transcribe(audio_file_path)
    assert isinstance(transcription, str)


# Test successful transcription with audio dict
def test_transcribe_with_audio_dict(whisper_model, audio_dict):
    transcription = whisper_model.transcribe(audio_dict)
    assert isinstance(transcription, str)


# Test for file not found error
def test_file_not_found(whisper_model, invalid_audio_file_path):
    with pytest.raises(Exception):
        whisper_model.transcribe(invalid_audio_file_path)


# Asynchronous tests
@pytest.mark.asyncio
async def test_async_transcription_success(
    whisper_model, audio_file_path
):
    transcription = await whisper_model.async_transcribe(
        audio_file_path
    )
    assert isinstance(transcription, str)


@pytest.mark.asyncio
async def test_async_transcription_failure(
    whisper_model, invalid_audio_file_path
):
    with pytest.raises(Exception):
        await whisper_model.async_transcribe(invalid_audio_file_path)


# Testing real-time transcription simulation
def test_real_time_transcription(
    whisper_model, audio_file_path, capsys
):
    whisper_model.real_time_transcribe(
        audio_file_path, chunk_duration=1
    )
    captured = capsys.readouterr()
    assert "Starting real-time transcription..." in captured.out


# Testing retry decorator for asynchronous function
@pytest.mark.asyncio
async def test_async_retry():
    @async_retry(max_retries=2, exceptions=(ValueError,), delay=0)
    async def failing_func():
        raise ValueError("Test")

    with pytest.raises(ValueError):
        await failing_func()


# Mocking the actual model to avoid GPU/CPU intensive operations during test
@pytest.fixture
def mocked_model(monkeypatch):
    model_mock = AsyncMock(AutoModelForSpeechSeq2Seq)
    processor_mock = MagicMock(AutoProcessor)
    monkeypatch.setattr(
        "swarms.models.distilled_whisperx.AutoModelForSpeechSeq2Seq.from_pretrained",
        model_mock,
    )
    monkeypatch.setattr(
        "swarms.models.distilled_whisperx.AutoProcessor.from_pretrained",
        processor_mock,
    )
    return model_mock, processor_mock


@pytest.mark.asyncio
async def test_async_transcribe_with_mocked_model(
    mocked_model, audio_file_path
):
    model_mock, processor_mock = mocked_model
    # Set up what the mock should return when it's called
    model_mock.return_value.generate.return_value = torch.tensor(
        [[0]]
    )
    processor_mock.return_value.batch_decode.return_value = [
        "mocked transcription"
    ]
    model_wrapper = DistilWhisperModel()
    transcription = await model_wrapper.async_transcribe(
        audio_file_path
    )
    assert transcription == "mocked transcription"
