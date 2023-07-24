import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class HuggingFaceLLM:
    """
    A class that represents a Language Model (LLM) powered by HuggingFace Transformers.

    Attributes:
        model_id (str): ID of the pre-trained model in HuggingFace Model Hub.
        device (str): Device to load the model onto.
        max_length (int): Maximum length of the generated sequence.
        tokenizer: Instance of the tokenizer corresponding to the model.
        model: The loaded model instance.
        logger: Logger instance for the class.
    """
    def __init__(self, model_id: str, device: str = None, max_length: int = 20, quantize: bool = False, quantization_config: dict = None):
        """
        Constructs all the necessary attributes for the HuggingFaceLLM object.

        Args:
            model_id (str): ID of the pre-trained model in HuggingFace Model Hub.
            device (str, optional): Device to load the model onto. Defaults to GPU if available, else CPU.
            max_length (int, optional): Maximum length of the generated sequence. Defaults to 20.
            quantize (bool, optional): Whether to apply quantization to the model. Defaults to False.
            quantization_config (dict, optional): Configuration for model quantization. Defaults to None, 
                and a standard configuration will be used if quantize is True.
        """
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
        """
        Generates text based on the input prompt using the loaded model.

        Args:
            prompt_text (str): Input prompt to generate text from.
            max_length (int, optional): Maximum length of the generated sequence. Defaults to None,
                and the max_length set during initialization will be used.

        Returns:
            str: Generated text.
        """
        max_length = max_length if max_length else self.max_length
        try:
            inputs = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs, max_length=max_length, do_sample=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Failed to generate the text: {e}")
            raise
