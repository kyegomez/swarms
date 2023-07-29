import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class HuggingFaceLLM:
    def __init__(self, model_id: str, device: str = None, max_length: int = 20, quantize: bool = False, quantization_config: dict = None):
        self.logger = logging.getLogger(__name__)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_id = model_id
        self.max_length = max_length

        bnb_config = None
        if quantize:
            if not quantization_config:
                quantization_config = {
                    'load_in_4bit': True,
                    'bnb_4bit_use_double_quant': True,
                    'bnb_4bit_quant_type': "nf4",
                    'bnb_4bit_compute_dtype': torch.bfloat16
                }
            bnb_config = BitsAndBytesConfig(**quantization_config)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb_config)
            self.model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load the model or the tokenizer: {e}")
            raise
    def generate_text(self, prompt_text: str, max_length: int = None):
        max_length = max_length if max_length else self.max_length
        try:
            inputs = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs, max_length=max_length, do_sample=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise
