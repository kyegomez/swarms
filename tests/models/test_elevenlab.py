import pytest
from unittest.mock import patch, mock_open
from swarms.models.eleven_labs import (
    ElevenLabsText2SpeechTool,
    ElevenLabsModel,
)
import os
from dotenv import load_dotenv

load_dotenv()

# Define some test data
SAMPLE_TEXT = "Hello, this is a test."
API_KEY = os.environ.get("ELEVEN_API_KEY")
EXPECTED_SPEECH_FILE = "expected_speech.wav"


@pytest.fixture
def eleven_labs_tool():
    return ElevenLabsText2SpeechTool()


# Basic functionality tests
def test_run_text_to_speech(eleven_labs_tool):
    speech_file = eleven_labs_tool.run(SAMPLE_TEXT)
    assert isinstance(speech_file, str)
    assert speech_file.endswith(".wav")


def test_play_speech(eleven_labs_tool):
    with patch(
        "builtins.open", mock_open(read_data="fake_audio_data")
    ):
        eleven_labs_tool.play(EXPECTED_SPEECH_FILE)


def test_stream_speech(eleven_labs_tool):
    with patch(
        "tempfile.NamedTemporaryFile", mock_open()
    ) as mock_file:
        eleven_labs_tool.stream_speech(SAMPLE_TEXT)
        mock_file.assert_called_with(
            mode="bx", suffix=".wav", delete=False
        )


# Testing fixture and environment variables
def test_api_key_validation(eleven_labs_tool):
    with patch(
        "langchain.utils.get_from_dict_or_env", return_value=API_KEY
    ):
        values = {"eleven_api_key": None}
        validated_values = eleven_labs_tool.validate_environment(
            values
        )
        assert "eleven_api_key" in validated_values


# Mocking the external library
def test_run_text_to_speech_with_mock(eleven_labs_tool):
    with patch(
        "tempfile.NamedTemporaryFile", mock_open()
    ) as mock_file, patch(
        "your_module._import_elevenlabs"
    ) as mock_elevenlabs:
        mock_elevenlabs_instance = mock_elevenlabs.return_value
        mock_elevenlabs_instance.generate.return_value = (
            b"fake_audio_data"
        )
        eleven_labs_tool.run(SAMPLE_TEXT)
        assert mock_file.call_args[1]["suffix"] == ".wav"
        assert mock_file.call_args[1]["delete"] is False
        assert mock_file().write.call_args[0][0] == b"fake_audio_data"


# Exception testing
def test_run_text_to_speech_error_handling(eleven_labs_tool):
    with patch("your_module._import_elevenlabs") as mock_elevenlabs:
        mock_elevenlabs_instance = mock_elevenlabs.return_value
        mock_elevenlabs_instance.generate.side_effect = Exception(
            "Test Exception"
        )
        with pytest.raises(
            RuntimeError,
            match=(
                "Error while running ElevenLabsText2SpeechTool: Test"
                " Exception"
            ),
        ):
            eleven_labs_tool.run(SAMPLE_TEXT)


# Parameterized testing
@pytest.mark.parametrize(
    "model",
    [ElevenLabsModel.MULTI_LINGUAL, ElevenLabsModel.MONO_LINGUAL],
)
def test_run_text_to_speech_with_different_models(
    eleven_labs_tool, model
):
    eleven_labs_tool.model = model
    speech_file = eleven_labs_tool.run(SAMPLE_TEXT)
    assert isinstance(speech_file, str)
    assert speech_file.endswith(".wav")
