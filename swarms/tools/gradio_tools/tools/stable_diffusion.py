from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Tuple

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool

if TYPE_CHECKING:
    import gradio as gr


class StableDiffusionTool(GradioTool):
    """Tool for calling stable diffusion from llm"""

    def __init__(
        self,
        name="StableDiffusion",
        description=(
            "An image generator. Use this to generate images based on "
            "text input. Input should be a description of what the image should "
            "look like. The output will be a path to an image file."
        ),
        src="gradio-client-demos/stable-diffusion",
        hf_token=None,
        duplicate=False,
    ) -> None:
        super().__init__(name, description, src, hf_token, duplicate)

    def create_job(self, query: str) -> Job:
        return self.client.submit(query, "", 9, fn_index=1)

    def postprocess(self, output: Tuple[Any] | Any) -> str:
        assert isinstance(output, str)
        return [
            os.path.join(output, i)
            for i in os.listdir(output)
            if not i.endswith("json")
        ][0]

    def _block_input(self, gr) -> "gr.components.Component":
        return gr.Textbox()

    def _block_output(self, gr) -> "gr.components.Component":
        return gr.Image()
