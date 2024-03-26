from transformers import AutoModelForCausalLM, AutoTokenizer
from swarms.models.base_llm import AbstractLLM


class Petals(AbstractLLM):
    """Petals Bloom models."""

    def __init__(
        self,
        model_name="bigscience/bloom-petals",
        temperature=0.7,
        max_new_tokens=256,
        top_p=0.9,
        top_k=None,
        do_sample=True,
        max_length=None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def _default_params(self):
        """Get the default parameters for calling Petals API."""
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "max_length": self.max_length,
        }

    def __call__(self, prompt):
        """Generate text using the Petals API."""
        params = self._default_params()
        inputs = self.tokenizer(prompt, return_tensors="pt")[
            "input_ids"
        ]
        outputs = self.model.generate(inputs, **params)
        return self.tokenizer.decode(outputs[0])
