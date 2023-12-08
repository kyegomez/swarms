from __future__ import annotations

from typing import TYPE_CHECKING

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool

if TYPE_CHECKING:
    import gradio as gr


class SAMImageSegmentationTool(GradioTool):
    """Tool for segmenting images based on natural language queries."""

    def __init__(
        self,
        name="SAMImageSegmentation",
        description=(
            "A tool for identifying objects in images. "
            "Input will be a five strings separated by a |: "
            "the first will be the full path or URL to an image file. "
            "The second will be the string query describing the objects to identify in the image. "
            "The query string should be as detailed as possible. "
            "The third will be the predicted_iou_threshold, if not specified by the user set it to 0.9. "
            "The fourth will be the stability_score_threshold, if not specified by the user set it to 0.8. "
            "The fifth is the clip_threshold, if not specified by the user set it to 0.85. "
            "The output will the a path with an image file with the identified objects overlayed in the image."
        ),
        src="curt-park/segment-anything-with-clip",
        hf_token=None,
        duplicate=False,
    ) -> None:
        super().__init__(name, description, src, hf_token, duplicate)

    def create_job(self, query: str) -> Job:
        try:
            (
                image,
                query,
                predicted_iou_threshold,
                stability_score_threshold,
                clip_threshold,
            ) = query.split("|")
        except ValueError as e:
            raise ValueError(
                "Not enough arguments passed to the SAMImageSegmentationTool! "
                "Expected 5 (image, query, predicted_iou_threshold, stability_score_threshold, clip_threshold)"
            ) from e
        return self.client.submit(
            float(predicted_iou_threshold),
            float(stability_score_threshold),
            float(clip_threshold),
            image,
            query.strip(),
            api_name="/predict",
        )

    def postprocess(self, output: str) -> str:
        return output

    def _block_input(self, gr) -> "gr.components.Component":
        return gr.Textbox()

    def _block_output(self, gr) -> "gr.components.Component":
        return gr.Audio()
