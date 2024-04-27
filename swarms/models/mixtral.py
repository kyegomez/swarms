from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from swarms.models.base_llm import BaseLLM


class Mixtral(BaseLLM):
    """Mixtral model.

    Args:
        model_name (str): The name or path of the pre-trained Mixtral model.
        max_new_tokens (int): The maximum number of new tokens to generate.
        *args: Variable length argument list.


    Examples:
        >>> from swarms.models import Mixtral
        >>> mixtral = Mixtral()
        >>> mixtral.run("Test task")
        'Generated text'
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-v0.1",
        max_new_tokens: int = 500,
        *args,
        **kwargs,
    ):
        """
        Initializes a Mixtral model.

        Args:
            model_name (str): The name or path of the pre-trained Mixtral model.
            max_new_tokens (int): The maximum number of new tokens to generate.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, *args, **kwargs
        )

    def run(self, task: Optional[str] = None, **kwargs):
        """
        Generates text based on the given task.

        Args:
            task (str, optional): The task or prompt for text generation.

        Returns:
            str: The generated text.
        """
        try:
            inputs = self.tokenizer(task, return_tensors="pt")

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                **kwargs,
            )

            out = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            return out
        except Exception as error:
            print(f"There is an error: {error} in Mixtral model.")
            raise error
