#!/usr/bin/env python
# coding=utf-8
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import torch
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer


class CpmBeeLLM(LLM):

    model_name : str = ""
    config: CPMBeeConfig = None
    tokenizer: CPMBeeTokenizer = None
    model: CPMBeeTorch = None
    
    def __init__(self, config_path: str, ckpt_path: str, device: str="cuda") -> None:
        super().__init__()
        self.model_name = ckpt_path
        self.config = CPMBeeConfig.from_json_file(config_path)
        self.tokenizer = CPMBeeTokenizer()
        self.model = CPMBeeTorch(config=self.config)

        self.model.load_state_dict(torch.load(ckpt_path))
        if device == "cuda":
            self.model.cuda()

    @property
    def _llm_type(self) -> str:
        return self.model_name
    
    def _call(self, prompt, stop: Optional[List[str]] = None) -> str:
        # use beam search
        beam_search = CPMBeeBeamSearch(
            model=self.model,
            tokenizer=self.tokenizer,
        )
        inference_results = beam_search.generate([{"source":prompt, "<ans>":""}], max_length=512, repetition_penalty=1.2, beam_size=1)
        output = inference_results[0]["<ans>"]
        return output
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}
    
if __name__ == "__main__":
    llm = CpmBeeLLM(config_path="path/to/cpm-bee/config.json", ckpt_path="path/to/cpm-bee/checkpoint/")
    print(llm("You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: What's the weather in Shanghai today? Should I bring an umbrella?, The last completed task has the result: According to the weather report, it is sunny in Shanghai today and there is no precipitation, so you do not need to bring an umbrella.. This result was based on this task description: Make a todo list about this objective: What's the weather in Shanghai today? Should I bring an umbrella?. These are incomplete tasks: . Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Do not generate repetitive tasks (e.g., tasks that have already been completed). If there is not futher task needed to complete the objective, only return NO TASK. Now return the tasks as an array."))
    
