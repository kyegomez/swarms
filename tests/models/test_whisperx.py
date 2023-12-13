import os
import subprocess
import tempfile
from unittest.mock import patch

import pytest
import whisperx
from pydub import AudioSegment
from pytube import YouTube
from swarms.models.whisperx_model import WhisperX


# Fixture to create a temporary directory for testing
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir


# Mock subprocess.run to prevent actual installation during tests
@patch.object(subprocess, "run")
def test_speech_to_text_install(mock_run):
    stt = WhisperX("https://www.youtube.com/watch?v=MJd6pr16LRM")
    stt.install()
    mock_run.assert_called_with(["pip", "install", "whisperx"])


# Mock pytube.YouTube and pytube.Streams for download tests
@patch("pytube.YouTube")
@patch.object(YouTube, "streams")
def test_speech_to_text_download_youtube_video(
    mock_streams, mock_youtube, temp_dir
):
    # Mock YouTube and streams
    video_url = "https://www.youtube.com/watch?v=MJd6pr16LRM"
    mock_stream = mock_streams().filter().first()
    mock_stream.download.return_value = os.path.join(
        temp_dir, "video.mp4"
    )
    mock_youtube.return_value = mock_youtube
    mock_youtube.streams = mock_streams

    stt = WhisperX(video_url)
    audio_file = stt.download_youtube_video()

    assert os.path.exists(audio_file)
    assert audio_file.endswith(".mp3")


# Mock whisperx.load_model and whisperx.load_audio for transcribe tests
@patch("whisperx.load_model")
@patch("whisperx.load_audio")
@patch("whisperx.load_align_model")
@patch("whisperx.align")
@patch.object(whisperx.DiarizationPipeline, "__call__")
def test_speech_to_text_transcribe_youtube_video(
    mock_diarization,
    mock_align,
    mock_align_model,
    mock_load_audio,
    mock_load_model,
    temp_dir,
):
    # Mock whisperx functions
    mock_load_model.return_value = mock_load_model
    mock_load_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"text": "Hello, World!"}],
    }

    mock_load_audio.return_value = "audio_path"
    mock_align_model.return_value = (mock_align_model, "metadata")
    mock_align.return_value = {
        "segments": [{"text": "Hello, World!"}]
    }

    # Mock diarization pipeline
    mock_diarization.return_value = None

    video_url = "https://www.youtube.com/watch?v=MJd6pr16LRM/video"
    stt = WhisperX(video_url)
    transcription = stt.transcribe_youtube_video()

    assert transcription == "Hello, World!"


# More tests for different scenarios and edge cases can be added here.


# Test transcribe method with provided audio file
def test_speech_to_text_transcribe_audio_file(temp_dir):
    # Create a temporary audio file
    audio_file = os.path.join(temp_dir, "test_audio.mp3")
    AudioSegment.silent(duration=500).export(audio_file, format="mp3")

    stt = WhisperX("https://www.youtube.com/watch?v=MJd6pr16LRM")
    transcription = stt.transcribe(audio_file)

    assert transcription == ""


# Test transcribe method when Whisperx fails
@patch("whisperx.load_model")
@patch("whisperx.load_audio")
def test_speech_to_text_transcribe_whisperx_failure(
    mock_load_audio, mock_load_model, temp_dir
):
    # Mock whisperx functions to raise an exception
    mock_load_model.side_effect = Exception("Whisperx failed")
    mock_load_audio.return_value = "audio_path"

    stt = WhisperX("https://www.youtube.com/watch?v=MJd6pr16LRM")
    transcription = stt.transcribe("audio_path")

    assert transcription == "Whisperx failed"


# Test transcribe method with missing 'segments' key in Whisperx output
@patch("whisperx.load_model")
@patch("whisperx.load_audio")
@patch("whisperx.load_align_model")
@patch("whisperx.align")
@patch.object(whisperx.DiarizationPipeline, "__call__")
def test_speech_to_text_transcribe_missing_segments(
    mock_diarization,
    mock_align,
    mock_align_model,
    mock_load_audio,
    mock_load_model,
):
    # Mock whisperx functions to return incomplete output
    mock_load_model.return_value = mock_load_model
    mock_load_model.transcribe.return_value = {"language": "en"}

    mock_load_audio.return_value = "audio_path"
    mock_align_model.return_value = (mock_align_model, "metadata")
    mock_align.return_value = {}

    # Mock diarization pipeline
    mock_diarization.return_value = None

    stt = WhisperX("https://www.youtube.com/watch?v=MJd6pr16LRM")
    transcription = stt.transcribe("audio_path")

    assert transcription == ""


# Test transcribe method with Whisperx align failure
@patch("whisperx.load_model")
@patch("whisperx.load_audio")
@patch("whisperx.load_align_model")
@patch("whisperx.align")
@patch.object(whisperx.DiarizationPipeline, "__call__")
def test_speech_to_text_transcribe_align_failure(
    mock_diarization,
    mock_align,
    mock_align_model,
    mock_load_audio,
    mock_load_model,
):
    # Mock whisperx functions to raise an exception during align
    mock_load_model.return_value = mock_load_model
    mock_load_model.transcribe.return_value = {
        "language": "en",
        "segments": [{"text": "Hello, World!"}],
    }

    mock_load_audio.return_value = "audio_path"
    mock_align_model.return_value = (mock_align_model, "metadata")
    mock_align.side_effect = Exception("Align failed")

    # Mock diarization pipeline
    mock_diarization.return_value = None

    stt = WhisperX("https://www.youtube.com/watch?v=MJd6pr16LRM")
    transcription = stt.transcribe("audio_path")

    assert transcription == "Align failed"


# Test transcribe_youtube_video when Whisperx diarization fails
@patch("pytube.YouTube")
@patch.object(YouTube, "streams")
@patch("whisperx.DiarizationPipeline")
@patch("whisperx.load_audio")
@patch("whisperx.load_align_model")
@patch("whisperx.align")
def test_speech_to_text_transcribe_diarization_failure(
    mock_align,
    mock_align_model,
    mock_load_audio,
    mock_diarization,
    mock_streams,
    mock_youtube,
    temp_dir,
):
    # Mock YouTube and streams
    video_url = "https://www.youtube.com/watch?v=MJd6pr16LRM"
    mock_stream = mock_streams().filter().first()
    mock_stream.download.return_value = os.path.join(
        temp_dir, "video.mp4"
    )
    mock_youtube.return_value = mock_youtube
    mock_youtube.streams = mock_streams

    # Mock whisperx functions
    mock_load_audio.return_value = "audio_path"
    mock_align_model.return_value = (mock_align_model, "metadata")
    mock_align.return_value = {
        "segments": [{"text": "Hello, World!"}]
    }

    # Mock diarization pipeline to raise an exception
    mock_diarization.side_effect = Exception("Diarization failed")

    stt = WhisperX(video_url)
    transcription = stt.transcribe_youtube_video()

    assert transcription == "Diarization failed"


# Add more tests for other scenarios and edge cases as needed.
