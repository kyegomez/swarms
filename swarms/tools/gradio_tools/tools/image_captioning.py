from __future__ import annotations

from typing import TYPE_CHECKING

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool

if TYPE_CHECKING:
    import gradio as gr


class ImageCaptioningTool(GradioTool):
    """Tool for captioning images."""

    def __init__(
        self,
        name="ImageCaptioner",
        description=(
            "An image captioner. Use this to create a caption for an image. "
            "Input will be a path to an image file. "
            "The output will be a caption of that image."
        ),
        src="taesiri/BLIP-2",
        hf_token=None,
        duplicate=True,
    ) -> None:
        super().__init__(name, description, src, hf_token, duplicate)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query.strip("'"), "Beam Search", fn_index=0)

    def postprocess(self, output: str) -> str:
        return output  # type: ignore

    def _block_input(self, gr) -> "gr.components.Component":
        return gr.Image()

    def _block_output(self, gr) -> "gr.components.Component":
        return gr.Textbox()
