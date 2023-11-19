from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, Tuple, Union

import gradio_client as grc
import huggingface_hub
from gradio_client.client import Job
from gradio_client.utils import QueueError

try:
    import langchain as lc

    LANGCHAIN_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    LANGCHAIN_INSTALLED = False


class GradioTool:
    def __init__(
        self,
        name: str,
        description: str,
        src: str,
        hf_token: str | None = None,
        duplicate: bool = True,
    ) -> None:
        self.name = name
        self.description = description
        if hf_token and self._is_space(src) and duplicate:
            self.client = grc.Client.duplicate(from_id=src, hf_token=hf_token)
            self.src = self.client.space_id
        else:
            self.src = src
            self.client = grc.Client(self.src, hf_token=hf_token)
        self.description += (
            "If the output is the string QUEUE_FULL you "
            "should try this action again and never stop until you "
            "don't see this message."
        )
        self._block = None

    @staticmethod
    def _is_space(src: str) -> bool:
        try:
            huggingface_hub.get_space_runtime(src)
            return True
        except huggingface_hub.hf_api.RepositoryNotFoundError:
            return False

    @abstractmethod
    def create_job(self, query: str) -> Job:
        pass

    @abstractmethod
    def postprocess(self, output: Union[Tuple[Any], Any]) -> str:
        pass

    def run(self, query: str):
        job = self.create_job(query)
        while not job.done():
            status = job.status()
            print(f"\nJob Status: {str(status.code)} eta: {status.eta}")
            time.sleep(30)
        try:
            output = self.postprocess(job.result())
        except QueueError:
            output = "QUEUE_FULL"
        return output

    # Optional gradio functionalities
    def _block_input(self, gr) -> "gr.components.Component":
        return gr.Textbox()

    def _block_output(self, gr) -> "gr.components.Component":
        return gr.Textbox()

    def block_input(self) -> "gr.components.Component":
        try:
            import gradio as gr

            GRADIO_INSTALLED = True
        except (ModuleNotFoundError, ImportError):
            GRADIO_INSTALLED = False
        if not GRADIO_INSTALLED:
            raise ModuleNotFoundError("gradio must be installed to call block_input")
        else:
            return self._block_input(gr)

    def block_output(self) -> "gr.components.Component":
        try:
            import gradio as gr

            GRADIO_INSTALLED = True
        except (ModuleNotFoundError, ImportError):
            GRADIO_INSTALLED = False
        if not GRADIO_INSTALLED:
            raise ModuleNotFoundError("gradio must be installed to call block_output")
        else:
            return self._block_output(gr)

    def block(self):
        """Get the gradio Blocks of this tool for visualization."""
        try:
            import gradio as gr
        except (ModuleNotFoundError, ImportError):
            raise ModuleNotFoundError("gradio must be installed to call block")
        if not self._block:
            self._block = gr.load(name=self.src, src="spaces")
        return self._block

    # Optional langchain functionalities
    @property
    def langchain(self) -> "langchain.agents.Tool":  # type: ignore
        if not LANGCHAIN_INSTALLED:
            raise ModuleNotFoundError(
                "langchain must be installed to access langchain tool"
            )

        return lc.agents.Tool(  # type: ignore
            name=self.name, func=self.run, description=self.description
        )

    def __repr__(self) -> str:
        return f"GradioTool(name={self.name}, src={self.src})"
