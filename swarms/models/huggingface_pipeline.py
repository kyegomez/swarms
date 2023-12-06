from abc import abstractmethod
from termcolor import colored
import torch

from swarms.models.base_llm import AbstractLLM

if torch.cuda.is_available():
    try:
        from optimum.nvidia.pipelines import pipeline
    except ImportError:
        from transformers.pipelines import pipeline


class HuggingfacePipeline(AbstractLLM):
    """HuggingfacePipeline



    Args:
        AbstractLLM (AbstractLLM): [description]
        task (str, optional): [description]. Defaults to "text-generation".
        model_name (str, optional): [description]. Defaults to None.
        use_fp8 (bool, optional): [description]. Defaults to False.
        *args: [description]
        **kwargs: [description]

    Raises:

    """

    def __init__(
        self,
        task: str = "text-generation",
        model_name: str = None,
        use_fp8: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pipe = pipeline(
            task, model_name, use_fp8=use_fp8 * args, **kwargs
        )

    @abstractmethod
    def run(self, task: str, *args, **kwargs):
        try:
            out = self.pipeline(task, *args, **kwargs)
            return out
        except Exception as e:
            print(
                colored(
                    f"Error in {self.__class__.__name__} pipeline",
                    "red",
                )
            )
