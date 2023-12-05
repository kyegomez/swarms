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
        model_name: str = "HuggingFaceH4/zephyr-7b-alpha",
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        system_prompt: str = "You are a friendly chatbot who always responds in the style of a pirate",
        max_new_tokens: int = 300,
        temperature: float = 0.5,
        top_k: float = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.messages = [
            {
                "role": "system",
                "content": f"{self.system_prompt}\n\nUser:",
            },
        ]

    def __call__(self, task: str):
        """Call the model"""
        prompt = self.pipe.tokenizer.apply_chat_template(
            self.messages,
            tokenize=self.tokenize,
            add_generation_prompt=self.add_generation_prompt,
        )
        outputs = self.pipe(
            prompt
        )  # max_new_token=self.max_new_tokens)
        print(outputs[0]["generated_text"])

    def chat(self, message: str):
        """
        Adds a user message to the conversation and generates a chatbot response.
        """
        # Add the user message to the conversation
        self.messages.append({"role": "user", "content": message})

        # Apply the chat template to format the messages
        prompt = self.pipe.tokenizer.apply_chat_template(
            self.messages,
            tokenize=self.tokenize,
            add_generation_prompt=self.add_generation_prompt,
        )

        # Generate a response
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        # Extract the generated text
        generated_text = outputs[0]["generated_text"]

        # Optionally, you could also add the chatbot's response to the messages list
        # However, the below line should be adjusted to extract the chatbot's response only
        # self.messages.append({"role": "bot", "content": generated_text})
        return generated_text
