# Hugging Face Custom Base LLM in Agent

Wrap a Hugging Face model as a custom LLM and pass it to the `Agent` class.

```bash
pip install swarms transformers torch
```

The `Agent` class expects a custom LLM with a `run(task: str) -> str` method.

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
        inputs = self.tokenizer(task, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __call__(self, task: str) -> str:
        return self.run(task)


llm = HuggingFaceLLM(model_name="meta-llama/Llama-2-7b-chat-hf")

agent = Agent(
    agent_name="HuggingFace-Agent",
    llm=llm,
    max_loops=1,
)

agent.run("Explain quantum computing in simple terms.")
```

Any Hugging Face causal language model can be used by changing `model_name`.
