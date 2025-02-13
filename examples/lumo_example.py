import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
)

from swarms import Agent


class Lumo:
    """
    A class for generating text using the Lumo model with 4-bit quantization.
    """

    def __init__(self):
        """
        Initializes the Lumo model with 4-bit quantization and a tokenizer.
        """
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.model = LlamaForCausalLM.from_pretrained(
            "lumolabs-ai/Lumo-70B-Instruct",
            device_map="auto",
            quantization_config=bnb_config,
            use_cache=False,
            attn_implementation="sdpa",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "lumolabs-ai/Lumo-70B-Instruct"
        )

    def run(self, task: str) -> str:
        """
        Generates text based on the given prompt using the Lumo model.

        Args:
            prompt (str): The input prompt for the model.

        Returns:
            str: The generated text.
        """
        inputs = self.tokenizer(task, return_tensors="pt").to(
            self.model.device
        )
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )


Agent(
    agent_name="Solana-Analysis-Agent",
    model_name=Lumo(),
    max_loops="auto",
    interactive=True,
    streaming_on=True,
).run("How do i create a smart contract in solana?")
