import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Mistral:
    """
    Mistral

    model = MistralWrapper(device="cuda", use_flash_attention=True, temperature=0.7, max_length=200)
    task = "My favourite condiment is"
    result = model.run(task)
    print(result)
    """
    def __init__(
        self, 
        model_name: str ="mistralai/Mistral-7B-v0.1", 
        device: str ="cuda", 
        use_flash_attention: bool = False,
        temperature: float = 1.0,
        max_length: int = 100,
        do_sample: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.use_flash_attention = use_flash_attention
        self.temperature = temperature
        self.max_length = max_length

        # Check if the specified device is available
        if not torch.cuda.is_available() and device == "cuda":
            raise ValueError("CUDA is not available. Please choose a different device.")

        # Load the model and tokenizer
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
        except Exception as e:
            raise ValueError(f"Error loading the Mistral model: {str(e)}")

    def run(
        self, 
        task: str
    ):
        """Run the model on a given task."""

        try:
            model_inputs = self.tokenizer(
                [task], 
                return_tensors="pt"
            ).to(self.device)
            generated_ids = self.model.generate(
                **model_inputs, 
                max_length=self.max_length, 
                do_sample=self.do_sample, 
                temperature=self.temperature,
                max_new_tokens=self.max_length
            )
            output_text = self.tokenizer.batch_decode(generated_ids)[0]
            return output_text
        except Exception as e:
            raise ValueError(f"Error running the model: {str(e)}")