"""
LayoutLMDocumentQA is a multimodal good for
visual question answering on real world docs lik invoice, pdfs, etc
"""

from transformers import pipeline

from swarms.models.base_multimodal_model import BaseMultiModalModel


class LayoutLMDocumentQA(BaseMultiModalModel):
    """
    LayoutLMDocumentQA for document question answering:

    Args:
        model_name (str, optional): [description]. Defaults to "impira/layoutlm-document-qa".
        task (str, optional): [description]. Defaults to "document-question-answering".

    Usage:
    >>> from swarms.models import LayoutLMDocumentQA
    >>> model = LayoutLMDocumentQA()
    >>> out = model("What is the total amount?", "path/to/img.png")
    >>> print(out)

    """

    def __init__(
        self,
        model_name: str = "impira/layoutlm-document-qa",
        task_type: str = "document-question-answering",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.task_type = task_type
        self.pipeline = pipeline(task_type, model=model_name)

    def __call__(self, task: str, img_path: str, *args, **kwargs):
        """Call the LayoutLMDocumentQA model

        Args:
            task (str): _description_
            img_path (str): _description_

        Returns:
            _type_: _description_
        """
        out = self.pipeline(img_path, task)
        out = str(out)
        return out
