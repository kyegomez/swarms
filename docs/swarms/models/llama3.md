## Llava3


```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from swarms.models.base_llm import BaseLLM


class Llama3(BaseLLM):
    """
    Llama3 class represents a Llama model for natural language generation.

        Args:
            model_id (str): The ID of the Llama model to use.
            system_prompt (str): The system prompt to use for generating responses.
            temperature (float): The temperature value for controlling the randomness of the generated responses.
            top_p (float): The top-p value for controlling the diversity of the generated responses.
            max_tokens (int): The maximum number of tokens to generate in the response.
            **kwargs: Additional keyword arguments.

        Attributes:
            model_id (str): The ID of the Llama model being used.
            system_prompt (str): The system prompt for generating responses.
            temperature (float): The temperature value for generating responses.
            top_p (float): The top-p value for generating responses.
            max_tokens (int): The maximum number of tokens to generate in the response.
            tokenizer (AutoTokenizer): The tokenizer for the Llama model.
            model (AutoModelForCausalLM): The Llama model for generating responses.

        Methods:
            run(task, *args, **kwargs): Generates a response for the given task.

    """

    def __init__(
        self,
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        system_prompt: str = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 4000,
        **kwargs,
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def run(self, task: str, *args, **kwargs):
        """
        Generates a response for the given task.

        Args:
            task (str): The user's task or input.

        Returns:
            str: The generated response.

        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            *args,
            **kwargs,
        )
        response = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(
            response, skip_special_tokens=True
        )
```