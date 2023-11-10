import os
import tempfile
from functools import wraps
from unittest.mock import patch

import numpy as np
import pytest
import torch

from swarms.models.distill_whisperx import DistilWhisperModel, async_retry


@pytest.fixture
def distil_whisper_model():
    return DistilWhisperModel()


def create_audio_file(data: np.ndarray, sample_rate: int, file_path: str):
    data.tofile(file_path)
    return file_path


def test_initialization(distil_whisper_model):
    assert isinstance(distil_whisper_model, DistilWhisperModel)
    assert isinstance(distil_whisper_model.model, torch.nn.Module)
    assert isinstance(distil_whisper_model.processor, torch.nn.Module)
    assert distil_whisper_model.device in ["cpu", "cuda:0"]


def test_transcribe_audio_file(distil_whisper_model):
    test_data = np.random.rand(16000)  # Simulated audio data (1 second)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file_path = create_audio_file(test_data, 16000, audio_file.name)
        transcription = distil_whisper_model.transcribe(audio_file_path)
        os.remove(audio_file_path)

    assert isinstance(transcription, str)
    assert transcription.strip() != ""


@pytest.mark.asyncio
async def test_async_transcribe_audio_file(distil_whisper_model):
    test_data = np.random.rand(16000)  # Simulated audio data (1 second)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file_path = create_audio_file(test_data, 16000, audio_file.name)
        transcription = await distil_whisper_model.async_transcribe(audio_file_path)
        os.remove(audio_file_path)

    assert isinstance(transcription, str)
    assert transcription.strip() != ""


def test_transcribe_audio_data(distil_whisper_model):
    test_data = np.random.rand(16000)  # Simulated audio data (1 second)
    transcription = distil_whisper_model.transcribe(test_data.tobytes())

    assert isinstance(transcription, str)
    assert transcription.strip() != ""


@pytest.mark.asyncio
async def test_async_transcribe_audio_data(distil_whisper_model):
    test_data = np.random.rand(16000)  # Simulated audio data (1 second)
    transcription = await distil_whisper_model.async_transcribe(test_data.tobytes())

    assert isinstance(transcription, str)
    assert transcription.strip() != ""


def test_real_time_transcribe(distil_whisper_model, capsys):
    test_data = np.random.rand(16000 * 5)  # Simulated audio data (5 seconds)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file_path = create_audio_file(test_data, 16000, audio_file.name)

        distil_whisper_model.real_time_transcribe(audio_file_path, chunk_duration=1)

        os.remove(audio_file_path)

    captured = capsys.readouterr()
    assert "Starting real-time transcription..." in captured.out
    assert "Chunk" in captured.out


def test_real_time_transcribe_audio_file_not_found(distil_whisper_model, capsys):
    audio_file_path = "non_existent_audio.wav"
    distil_whisper_model.real_time_transcribe(audio_file_path, chunk_duration=1)

    captured = capsys.readouterr()
    assert "The audio file was not found." in captured.out


@pytest.fixture
def mock_async_retry():
    def _mock_async_retry(retries=3, exceptions=(Exception,), delay=1):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    with patch("distil_whisper_model.async_retry", new=_mock_async_retry()):
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
    decorated_function = async_retry(max_retries=2)(mock_async_function)
    result = await decorated_function()
    assert result == "Success"


def test_create_audio_file():
    test_data = np.random.rand(16000)  # Simulated audio data (1 second)
    sample_rate = 16000
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file_path = create_audio_file(test_data, sample_rate, audio_file.name)

        assert os.path.exists(audio_file_path)
        os.remove(audio_file_path)


if __name__ == "__main__":
    pytest.main()
