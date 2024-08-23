import json
import os
from swarms import Agent, OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
import asyncio
from swarms.telemetry.async_log_telemetry import send_telemetry

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent-General-11",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=False,
    dashboard=False,
    verbose=True,
    # interactive=True, # Set to False to disable interactive mode
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    # tools=[#Add your functions here# ],
    # stopping_token="Stop!",
    # docs_folder="docs", # Enter your folder name
    # pdf_path="docs/finance_agent.pdf",
    # sop="Calculate the profit for a company.",
    # sop_list=["Calculate the profit for a company."],
    user_name="swarms_corp",
    # # docs="",
    retry_attempts=3,
    # context_length=1000,
    # tool_schema = dict
    context_length=200000,
    tool_system_prompt=None,
)

# # Convert the agent object to a dictionary
data = agent.to_dict()
data = json.dumps(data)


# Async
async def send_data():
    response_status, response_data = await send_telemetry(data)
    print(response_status, response_data)


# Run the async function
asyncio.run(send_data())
