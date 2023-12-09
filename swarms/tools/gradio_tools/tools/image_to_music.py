from typing import TYPE_CHECKING, Any, Tuple, Union

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool

if TYPE_CHECKING:
    import gradio as gr


class ImageToMusicTool(GradioTool):
    def __init__(
        self,
        name="ImagetoMusic",
        description=(
            "A tool for creating music from images. Use this tool to create a musical "
            "track from an image. Input will be a path to an image file. "
            "The output will be an audio file generated from that image."
        ),
        src="fffiloni/img-to-music",
        hf_token=None,
        duplicate=False,
    ) -> None:
        super().__init__(name, description, src, hf_token, duplicate)

    def create_job(self, query: str) -> Job:
        return self.client.submit(
            query.strip("'"), 15, "medium", "loop", None, fn_index=0
        )

    def postprocess(self, output: Union[Tuple[Any], Any]) -> str:
        return output[1]  # type: ignore

    def _block_input(self, gr) -> "gr.components.Component":
        return gr.Image()

    def _block_output(self, gr) -> "gr.components.Component":
        return gr.Audio()
