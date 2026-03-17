# Integrate Hugging Face with a Custom Base LLM in Agent

This example shows how to wrap a Hugging Face model as a custom LLM and pass it to the `Agent` class.

## Installation

```bash
pip install swarms transformers torch
```

## The `run` Method Contract

The `Agent` class expects a custom LLM object with a `run(task: str) -> str` method (and optionally a `__call__` method). The method receives the task string and must return the model's response as a string.

## Full Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from swarms import Agent


class HuggingFaceLLM:
    def __init__(self, model_name: str, max_tokens: int = 500):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.max_tokens = max_tokens

    def run(self, task: str) -> str:
        inputs = self.tokenizer(task, return_tensors="pt").to(
            self.model.device
        )
        outputs = self.model.generate(
            **inputs, max_new_tokens=self.max_tokens
        )
        return self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

    def __call__(self, task: str) -> str:
        return self.run(task)


llm = HuggingFaceLLM(model_name="meta-llama/Llama-2-7b-chat-hf")

agent = Agent(
    agent_name="HuggingFace-Agent",
    llm=llm,
    max_loops=1,
    agent_description="An agent powered by a local Hugging Face model",
)

agent.run("Explain quantum computing in simple terms.")
```

## How It Works

1. `HuggingFaceLLM.__init__` loads the tokenizer and model from the Hugging Face Hub.
2. `run(task)` tokenizes the input, generates output tokens, and decodes them back to a string.
3. The `llm` instance is passed to `Agent(llm=...)`. The agent calls `llm.run(task)` (or `llm(task)`) when processing each step.

Any Hugging Face causal language model can be used by changing `model_name`.
