from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class MultiModalLlava:
    """
    LLava Model

    Args:
        model_name_or_path: The model name or path to the model
        revision: The revision of the model to use
        device: The device to run the model on
        max_new_tokens: The maximum number of tokens to generate
        do_sample: Whether or not to use sampling
        temperature: The temperature of the sampling
        top_p: The top p value for sampling
        top_k: The top k value for sampling
        repetition_penalty: The repetition penalty for sampling
        device_map: The device map to use

    Methods:
        __call__: Call the model
        chat: Interactive chat in terminal

    Example:
        >>> from swarms.models.llava import LlavaModel
        >>> model = LlavaModel(device="cpu")
        >>> model("Hello, I am a robot.")
    """

    def __init__(
        self,
        model_name_or_path="TheBloke/llava-v1.5-13B-GPTQ",
        revision="main",
        device="cuda",
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1,
        device_map: str = "auto",
    ):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            trust_remote_code=False,
            revision=revision,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=0 if self.device == "cuda" else -1,
        )

    def __call__(self, prompt):
        """Call the model"""
        return self.pipe(prompt)[0]["generated_text"]

    def chat(self):
        """Interactive chat in terminal"""
        print(
            "Starting chat with LlavaModel. Type 'exit' to end the"
            " session."
        )
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            response = self(user_input)
            print(f"Model: {response}")
