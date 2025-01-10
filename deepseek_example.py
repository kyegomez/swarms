import os

from dotenv import load_dotenv
from openai import OpenAI

from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

load_dotenv()


class DeepSeekChat:
    def __init__(
        self,
        api_key: str = os.getenv("DEEPSEEK_API_KEY"),
        system_prompt: str = None,
    ):
        self.api_key = api_key

        self.client = OpenAI(
            api_key=api_key, base_url="https://api.deepseek.com"
        )

    def run(self, task: str):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant",
                },
                {"role": "user", "content": task},
            ],
            stream=False,
        )

        print(response)

        out = response.choices[0].message.content
        print(out)

        return out


model = DeepSeekChat()

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    max_loops=1,
    llm=model,
    dynamic_temperature_enabled=True,
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=8192,
    return_step_meta=False,
    output_type="str",  # "json", "dict", "csv" OR "string" "yaml" and
    auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
    max_tokens=4000,  # max output tokens
)

print(
    agent.run(
        "Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown.",
    )
)
