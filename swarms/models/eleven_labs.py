import tempfile
from enum import Enum
from typing import Any, Dict, Union

from langchain.utils import get_from_dict_or_env
from pydantic import model_validator

from swarms.tools.tool import BaseTool


def _import_elevenlabs() -> Any:
    try:
        import elevenlabs
    except ImportError as e:
        raise ImportError(
            "Cannot import elevenlabs, please install `pip install"
            " elevenlabs`."
        ) from e
    return elevenlabs


class ElevenLabsModel(str, Enum):
    """Models available for Eleven Labs Text2Speech."""

    MULTI_LINGUAL = "eleven_multilingual_v1"
    MONO_LINGUAL = "eleven_monolingual_v1"


class ElevenLabsText2SpeechTool(BaseTool):
    """Tool that queries the Eleven Labs Text2Speech API.

    In order to set this up, follow instructions at:
    https://docs.elevenlabs.io/welcome/introduction

    Attributes:
        model (ElevenLabsModel): The model to use for text to speech.
            Defaults to ElevenLabsModel.MULTI_LINGUAL.
        name (str): The name of the tool. Defaults to "eleven_labs_text2speech".
        description (str): The description of the tool.
            Defaults to "A wrapper around Eleven Labs Text2Speech. Useful for when you need to convert text to speech. It supports multiple languages, including English, German, Polish, Spanish, Italian, French, Portuguese, and Hindi."


    Usage:
    >>> from swarms.models import ElevenLabsText2SpeechTool
    >>> stt = ElevenLabsText2SpeechTool()
    >>> speech_file = stt.run("Hello world!")
    >>> stt.play(speech_file)
    >>> stt.stream_speech("Hello world!")

    """

    model: Union[ElevenLabsModel, str] = ElevenLabsModel.MULTI_LINGUAL

    name: str = "eleven_labs_text2speech"
    description: str = (
        "A wrapper around Eleven Labs Text2Speech. Useful for when"
        " you need to convert text to speech. It supports multiple"
        " languages, including English, German, Polish, Spanish,"
        " Italian, French, Portuguese, and Hindi. "
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        _ = get_from_dict_or_env(
            values, "eleven_api_key", "ELEVEN_API_KEY"
        )

        return values

    def _run(
        self,
        task: str,
    ) -> str:
        """Use the tool."""
        elevenlabs = _import_elevenlabs()
        try:
            speech = elevenlabs.generate(text=task, model=self.model)
            with tempfile.NamedTemporaryFile(
                mode="bx", suffix=".wav", delete=False
            ) as f:
                f.write(speech)
            return f.name
        except Exception as e:
            raise RuntimeError(
                f"Error while running ElevenLabsText2SpeechTool: {e}"
            )

    def play(self, speech_file: str) -> None:
        """Play the text as speech."""
        elevenlabs = _import_elevenlabs()
        with open(speech_file, mode="rb") as f:
            speech = f.read()

        elevenlabs.play(speech)

    def stream_speech(self, query: str) -> None:
        """Stream the text as speech as it is generated.
        Play the text in your speakers."""
        elevenlabs = _import_elevenlabs()
        speech_stream = elevenlabs.generate(
            text=query, model=self.model, stream=True
        )
        elevenlabs.stream(speech_stream)

    def save(self, speech_file: str, path: str) -> None:
        """Save the speech file to a path."""
        raise NotImplementedError(
            "Saving not implemented for this tool."
        )

    def __str__(self):
        return "ElevenLabsText2SpeechTool"
