import os

from dotenv import load_dotenv
from swarm_models import OpenAIChat

from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from new_features_examples.async_executor import HighSpeedExecutor

load_dotenv()

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    # autosave=True,
    # dashboard=False,
    # verbose=True,
    # dynamic_temperature_enabled=True,
    # saved_state_path="finance_agent.json",
    # user_name="swarms_corp",
    # retry_attempts=1,
    # context_length=200000,
    # return_step_meta=True,
    # output_type="json",  # "json", "dict", "csv" OR "string" soon "yaml" and
    # auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
    # # artifacts_on=True,
    # artifacts_output_path="roth_ira_report",
    # artifacts_file_extension=".txt",
    # max_tokens=8000,
    # return_history=True,
)


def execute_agent(
    task: str = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria. Create a report on this question.",
):
    return agent.run(task)


executor = HighSpeedExecutor()
results = executor.run(execute_agent, 2)

print(results)
