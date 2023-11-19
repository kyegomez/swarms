from typing import TYPE_CHECKING

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool

if TYPE_CHECKING:
    import gradio as gr


class ClipInterrogatorTool(GradioTool):
    def __init__(
        self,
        name="ClipInterrogator",
        description=(
            "A tool for reverse engineering a prompt from a source image. "
            "Use this tool to create a prompt for StableDiffusion that matches the "
            "input image. The imput is a path to an image. The output is a text string."
        ),
        src="pharma/CLIP-Interrogator",
        hf_token=None,
        duplicate=True,
    ) -> None:
        super().__init__(name, description, src, hf_token, duplicate)

    def create_job(self, query: str) -> Job:
        return self.client.submit(
            query, "ViT-L (best for Stable Diffusion 1.*)", "best", fn_index=3
        )

    def postprocess(self, output: str) -> str:
        return output

    def _block_input(self, gr) -> "gr.components.Component":
        return gr.Image()
