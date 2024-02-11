from typing import Optional

from modelscope import AutoModelForCausalLM, AutoTokenizer

from swarms.models.base_llm import AbstractLLM


class ModelScopeAutoModel(AbstractLLM):
    """
    ModelScopeAutoModel is a class that represents a model for generating text using the ModelScope framework.

    Args:
        model_name (str): The name or path of the pre-trained model.
        tokenizer_name (str, optional): The name or path of the tokenizer to use. Defaults to None.
        device (str, optional): The device to use for model inference. Defaults to "cuda".
        device_map (str, optional): The device mapping for multi-GPU setups. Defaults to "auto".
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 500.
        skip_special_tokens (bool, optional): Whether to skip special tokens during decoding. Defaults to True.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing input text.
        model (AutoModelForCausalLM): The pre-trained model for generating text.

    Methods:
        run(task, *args, **kwargs): Generates text based on the given task.

    Examples:
    >>> from swarms.models import ModelScopeAutoModel
    >>> mp = ModelScopeAutoModel(
    ...     model_name="gpt2",
    ... )
    >>> mp.run("Generate a 10,000 word blog on health and wellness.")
    """

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        device: str = "cuda",
        device_map: str = "auto",
        max_new_tokens: int = 500,
        skip_special_tokens: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.skip_special_tokens = skip_special_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=device_map * args, **kwargs
        )

    def run(self, task: str, *args, **kwargs):
        """
        Run the model on the given task.

        Parameters:
            task (str): The input task to be processed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated output from the model.
        """
        text = self.tokenizer(task, return_tensors="pt")

        outputs = self.model.generate(
            **text, max_new_tokens=self.max_new_tokens, **kwargs
        )

        return self.tokenizer.decode(
            outputs[0], skip_special_tokens=self.skip_special_tokens
        )
