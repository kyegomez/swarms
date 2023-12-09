from typing import TYPE_CHECKING

from gradio_client.client import Job

from gradio_tools.tools.gradio_tool import GradioTool

if TYPE_CHECKING:
    import gradio as gr


class DocQueryDocumentAnsweringTool(GradioTool):
    def __init__(
        self,
        name="DocQuery",
        description=(
            "A tool for answering questions about a document from the from the image of the document. Input will be a two strings separated by a comma: the first will be the path or URL to an image of a document. The second will be your question about the document."
            "The output will the text answer to your question."
        ),
        src="abidlabs/docquery",
        hf_token=None,
        duplicate=True,
    ) -> None:
        super().__init__(name, description, src, hf_token, duplicate)

    def create_job(self, query: str) -> Job:
        img, question = query.split(",")
        return self.client.submit(img.strip(), question.strip(), api_name="/predict")

    def postprocess(self, output: str) -> str:
        return output

    def _block_input(self, gr) -> "gr.components.Component":
        return gr.Image()
