from __future__ import annotations

from typing import TYPE_CHECKING

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool

if TYPE_CHECKING:
    import gradio as gr


class WhisperAudioTranscriptionTool(GradioTool):
    def __init__(
        self,
        name="WhisperAudioTranscription",
        description=(
            "A tool for transcribing audio. Use this tool to transcribe an audio file. "
            "track from an image. Input will be a path to an audio file. "
            "The output will the text transcript of that file."
        ),
        src="abidlabs/whisper",
        hf_token=None,
        duplicate=False,
    ) -> None:
        super().__init__(name, description, src, hf_token, duplicate)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query, api_name="/predict")

    def postprocess(self, output: str) -> str:
        return output

    def _block_input(self, gr) -> "gr.components.Component":
        return gr.Audio()
