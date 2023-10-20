"""Zephyr by HF"""
import torch
from transformers import pipeline


class Zephyr:
    """
    Zehpyr model from HF


    Args:
        max_new_tokens(int) = Number of max new tokens
        temperature(float) = temperature of the LLM
        top_k(float) = top k of the model set to 50
        top_p(float) = top_p of the model set to 0.95



    Usage:
    >>> model = Zephyr()
    >>> output = model("Generate hello world in python")


    """

    def __init__(
        self,
        max_new_tokens: int = 300,
        temperature: float = 0.5,
        top_k: float = 50,
        top_p: float = 0.95,
    ):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.pipe = pipeline(
            "text-generation",
            model="HuggingFaceH4/zephyr-7b-alpha",
            torch_dtype=torch.bfloa16,
            device_map="auto",
        )
        self.messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {
                "role": "user",
                "content": "How many helicopters can a human eat in one sitting?",
            },
        ]

    def __call__(self, text: str):
        """Call the model"""
        prompt = self.pipe.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(prompt, max_new_token=self.max_new_tokens)
        print(outputs[0])["generated_text"]
