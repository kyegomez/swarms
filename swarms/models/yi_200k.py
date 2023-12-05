from transformers import AutoModelForCausalLM, AutoTokenizer


class Yi34B200k:
    """
    A class for eaasy interaction with Yi34B200k

    Attributes:
    -----------
    model_id: str
        The model id of the model to be used.
    device_map: str
        The device to be used for inference.
    torch_dtype: str
        The torch dtype to be used for inference.
    max_length: int
        The maximum length of the generated text.
    repitition_penalty: float
        The repitition penalty to be used for inference.
    no_repeat_ngram_size: int
        The no repeat ngram size to be used for inference.
    temperature: float
        The temperature to be used for inference.

    Methods:
    --------
    __call__(self, task: str) -> str:
        Generates text based on the given prompt.


    """

    def __init__(
        self,
        model_id: str = "01-ai/Yi-34B-200K",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_length: int = 512,
        repitition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 5,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.8,
    ):
        super().__init__()
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_length = max_length
        self.repitition_penalty = repitition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

    def __call__(self, task: str):
        """
        Generates text based on the given prompt.

        Args:
            prompt (str): The input text prompt.
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        inputs = self.tokenizer(task, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids.cuda(),
            max_length=self.max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            repetition_penalty=self.repitition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        return self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )


# # Example usage
# yi34b = Yi34B200k()
# prompt = "There's a place where time stands still. A place of breathtaking wonder, but also"
# generated_text = yi34b(prompt)
# print(generated_text)
