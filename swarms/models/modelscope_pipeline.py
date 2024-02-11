from modelscope.pipelines import pipeline

from swarms.models.base_llm import AbstractLLM


class ModelScopePipeline(AbstractLLM):
    """
    A class representing a ModelScope pipeline.

    Args:
        type_task (str): The type of task for the pipeline.
        model_name (str): The name of the model for the pipeline.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        type_task (str): The type of task for the pipeline.
        model_name (str): The name of the model for the pipeline.
        model: The pipeline model.

    Methods:
        run: Runs the pipeline for a given task.

    Examples:
    >>> from swarms.models import ModelScopePipeline
    >>> mp = ModelScopePipeline(
    ...     type_task="text-generation",
    ...     model_name="gpt2",
    ... )
    >>> mp.run("Generate a 10,000 word blog on health and wellness.")

    """

    def __init__(
        self, type_task: str, model_name: str, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.type_task = type_task
        self.model_name = model_name

        self.model = pipeline(
            self.type_task, model=self.model_name, *args, **kwargs
        )

    def run(self, task: str, *args, **kwargs):
        """
        Runs the pipeline for a given task.

        Args:
            task (str): The task to be performed by the pipeline.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of running the pipeline on the given task.

        """
        return self.model(task, *args, **kwargs)
