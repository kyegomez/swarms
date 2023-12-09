#!/usr/bin/env python
# coding=utf-8
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class LoraModel(LLM):
    
    model_name: str = ""
    tokenizer: AutoTokenizer = None
    model: PeftModel = None
    use_gpu: bool = True

    def __init__(self, base_name_or_path: str, model_name_or_path: str, device: str="cuda", cpu_offloading: bool=False, load_8bit: bool=False) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_name_or_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            base_name_or_path,
            load_in_8bit=load_8bit,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(
            model,
            model_name_or_path
        )
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.use_gpu = (True if device == "cuda" else False)
        if (device == "cuda" and not cpu_offloading) or device == "mps":
            self.model.to(device)
        
    @property
    def _llm_type(self) -> str:
        return self.model_name
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(
            prompt,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        inputs_len = inputs["input_ids"].shape[1]
        generated_outputs = self.model.generate(
            input_ids=(inputs["input_ids"].cuda() if self.use_gpu else inputs["input_ids"]),
            attention_mask=(inputs["attention_mask"].cuda() if self.use_gpu else inputs["attention_mask"]),
            max_new_tokens=512,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        decoded_output = self.tokenizer.batch_decode(
            generated_outputs[..., inputs_len:], skip_special_tokens=True)
        output = decoded_output[0]
        return output
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}
    
if __name__ == "__main__":
    llm = LoraModel(base_name_or_path="huggyllama/llama-7b", model_name_or_path="pooruss-lsh/tool-llama7b-single-tool-lora")
    print(llm("You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: What's the weather in Shanghai today? Should I bring an umbrella?, The last completed task has the result: According to the weather report, it is sunny in Shanghai today and there is no precipitation, so you do not need to bring an umbrella.. This result was based on this task description: Make a todo list about this objective: What's the weather in Shanghai today? Should I bring an umbrella?. These are incomplete tasks: . Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Do not generate repetitive tasks (e.g., tasks that have already been completed). If there is not futher task needed to complete the objective, only return NO TASK. Now return the tasks as an array."))