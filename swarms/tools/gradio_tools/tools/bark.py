from __future__ import annotations

from typing import TYPE_CHECKING

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool

if TYPE_CHECKING:
    import gradio as gr

SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

SUPPORTED_LANGS = {lang: code for lang, code in SUPPORTED_LANGS}
VOICES = ["Unconditional", "Announcer"]
SUPPORTED_SPEAKERS = VOICES + [p for p in SUPPORTED_LANGS]

NON_SPEECH_TOKENS = [
    "[laughter]",
    "[laughs]",
    "[sighs]",
    "[music]",
    "[gasps]",
    "[clears throat]",
    "'♪' for song lyrics. Put ♪ on either side of the the text",
    "'…' for hesitations",
]


class BarkTextToSpeechTool(GradioTool):
    """Tool for calling bark text-to-speech llm."""

    def __init__(
        self,
        name="BarkTextToSpeech",
        description=(
            "A tool for text-to-speech. Use this tool to convert text "
            "into sounds that sound like a human read it. Input will be a two strings separated by a |: "
            "the first will be the text to read. The second will be the desired speaking language. "
            f"It MUST be one of the following choices {','.join(SUPPORTED_SPEAKERS)}. "
            f"Additionally, you can include the following non speech tokens: {NON_SPEECH_TOKENS}"
            "The output will the text transcript of that file."
        ),
        src="suno/bark",
        hf_token=None,
        duplicate=False,
    ) -> None:
        super().__init__(name, description, src, hf_token, duplicate)

    def create_job(self, query: str) -> Job:
        try:
            text, speaker = (
                query[: query.rindex("|")],
                query[(query.rindex("|") + 1) :].strip(),
            )
        except ValueError:
            text, speaker = query, "Unconditional"
        if speaker in VOICES:
            pass
        elif speaker in SUPPORTED_LANGS:
            speaker = f"Speaker 0 ({SUPPORTED_LANGS[speaker]})"
        else:
            speaker = "Unconditional"
        return self.client.submit(text, speaker, fn_index=3)

    def postprocess(self, output: str) -> str:
        return output

    def _block_input(self, gr) -> "gr.components.Component":
        return gr.Textbox()

    def _block_output(self, gr) -> "gr.components.Component":
        return gr.Audio()
