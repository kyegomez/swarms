from transformers import AutoModelForCausalLM, AutoTokenizer
from swarms.structs.agent import Agent


class BaseLLM:
    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] = [],
        system_prompt: str = "You are a base LLM agent.",
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.system_prompt = system_prompt

        model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"

        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def run(self, task: str, *args, **kwargs):
        # prepare the model input
        prompt = task
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.model.device
        )

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=self.max_tokens
        )
        output_ids = generated_ids[0][
            len(model_inputs.input_ids[0]) :
        ].tolist()

        content = self.tokenizer.decode(
            output_ids, skip_special_tokens=True
        )

        return content

    def __call__(self, task: str, *args, **kwargs):
        return self.run(task, *args, **kwargs)


agent = Agent(
    llm=BaseLLM(),
    agent_name="coder-agent",
    system_prompt="You are a coder agent.",
    dynamic_temperature_enabled=True,
    max_loops=2,
)
