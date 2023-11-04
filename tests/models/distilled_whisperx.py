# test_distilled_whisperx.py

from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from swarms.models.distilled_whisperx import DistilWhisperModel, async_retry


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
async def test_async_transcription_success(whisper_model, audio_file_path):
    transcription = await whisper_model.async_transcribe(audio_file_path)
    assert isinstance(transcription, str)


@pytest.mark.asyncio
async def test_async_transcription_failure(whisper_model, invalid_audio_file_path):
    with pytest.raises(Exception):
        await whisper_model.async_transcribe(invalid_audio_file_path)


# Testing real-time transcription simulation
def test_real_time_transcription(whisper_model, audio_file_path, capsys):
    whisper_model.real_time_transcribe(audio_file_path, chunk_duration=1)
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
        "swarms.models.distilled_whisperx.AutoProcessor.from_pretrained", processor_mock
    )
    return model_mock, processor_mock


@pytest.mark.asyncio
async def test_async_transcribe_with_mocked_model(mocked_model, audio_file_path):
    model_mock, processor_mock = mocked_model
    # Set up what the mock should return when it's called
    model_mock.return_value.generate.return_value = torch.tensor([[0]])
    processor_mock.return_value.batch_decode.return_value = ["mocked transcription"]
    model_wrapper = DistilWhisperModel()
    transcription = await model_wrapper.async_transcribe(audio_file_path)
    assert transcription == "mocked transcription"

