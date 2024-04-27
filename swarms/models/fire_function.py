import json
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from swarms.models.base_llm import BaseLLM


class FireFunctionCaller(BaseLLM):
    """
    A class that represents a caller for the FireFunction model.

    Args:
        model_name (str): The name of the model to be used.
        device (str): The device to be used.
        function_spec (Any): The specification of the function.
        max_tokens (int): The maximum number of tokens.
        system_prompt (str): The system prompt.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Methods:
        run(self, task: str, *args, **kwargs) -> None: Run the function with the given task and arguments.

    Examples:
        >>> fire_function_caller = FireFunctionCaller()
        >>> fire_function_caller.run("Add 2 and 3")
    """

    def __init__(
        self,
        model_name: str = "fireworks-ai/firefunction-v1",
        device: str = "cuda",
        function_spec: Any = None,
        max_tokens: int = 3000,
        system_prompt: str = "You are a helpful assistant with access to functions. Use them if required.",
        *args,
        **kwargs,
    ):
        super().__init__(model_name, device)
        self.model_name = model_name
        self.device = device
        self.fucntion_spec = function_spec
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", *args, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.functions = json.dumps(function_spec, indent=4)

    def run(self, task: str, *args, **kwargs):
        """
        Run the function with the given task and arguments.

        Args:
            task (str): The task to be performed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        messages = [
            {"role": "functions", "content": self.functions},
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": task,
            },
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=self.max_tokens,
            *args,
            **kwargs,
        )
        decoded = self.tokenizer.batch_decode(generated_ids)
        print(decoded[0])
