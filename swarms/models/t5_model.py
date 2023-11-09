#!/usr/bin/env python
# coding=utf-8
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from transformers import T5Tokenizer, T5ForConditionalGeneration


class T5Model(LLM):
    model_name: str = ""
    tokenizer: T5Tokenizer = None
    model: T5ForConditionalGeneration = None

    def __init__(self, huggingface_model_name: str) -> None:
        super().__init__()
        self.model_name = huggingface_model_name 
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name) 

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

        # inputs_len = inputs["input_ids"].shape[1]    

        generated_outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=512,
        )
        decoded_output = self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        output = decoded_output[0]
        return output


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}

if __name__ == "__main__":
    llm = T5Model("t5-small")
    # print(llm("translate English to German: The house is wonderful."))
    print(llm("You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: What's the weather in Shanghai today? Should I bring an umbrella?, The last completed task has the result: According to the weather report, it is sunny in Shanghai today and there is no precipitation, so you do not need to bring an umbrella.. This result was based on this task description: Make a todo list about this objective: What's the weather in Shanghai today? Should I bring an umbrella?. These are incomplete tasks: . Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Do not generate repetitive tasks (e.g., tasks that have already been completed). If there is not futher task needed to complete the objective, only return NO TASK. Now return the tasks as an array."))
