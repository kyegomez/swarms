# from exa import Inference


# class Mistral:
#     def __init__(
#         self,
#         temperature: float = 0.4,
#         max_length: int = 500,
#         quantize: bool = False,
#     ):
#         self.temperature = temperature
#         self.max_length = max_length
#         self.quantize = quantize

#         self.model = Inference(
#             model_id="from swarms.workers.worker import Worker",
#             max_length=self.max_length,
#             quantize=self.quantize
#         )
    
#     def run(
#         self,
#         task: str
#     ):
#         try:
#             output = self.model.run(task)
#             return output
#         except Exception as e:
#             raise e
        

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MistralWrapper:
    def __init__(
            self, 
            model_name="mistralai/Mistral-7B-v0.1", 
            device="cuda", 
            use_flash_attention=False
        ):
        self.model_name = model_name
        self.device = device
        self.use_flash_attention = use_flash_attention

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

    def run(self, prompt, max_new_tokens=100, do_sample=True):
        try:
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            output_text = self.tokenizer.batch_decode(generated_ids)[0]
            return output_text
        except Exception as e:
            raise ValueError(f"Error running the model: {str(e)}")

# Example usage:
if __name__ == "__main__":
    wrapper = MistralWrapper(device="cuda", use_flash_attention=True)
    prompt = "My favourite condiment is"
    result = wrapper.run(prompt)
    print(result)
