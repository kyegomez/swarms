from transformers import pipeline
from swarms import Agent

class GPTOSS:
    def __init__(
        self,
        model_id: str = "openai/gpt-oss-20b",
        max_new_tokens: int = 256,
        temperature: int = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.model_id = model_id

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
            temperature=temperature,
        )

    def run(self, task: str):
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        outputs = self.pipe(
            self.messages,
            max_new_tokens=self.max_new_tokens,
        )

        return outputs[0]["generated_text"][-1]

agent = Agent(
    name="GPT-OSS-Agent",
    llm=GPTOSS(),
    system_prompt="You are a helpful assistant.",
)

agent.run(task="Explain quantum mechanics clearly and concisely.")
