# ruff: noqa: E402 # Ignore module level import not at top of file

import time

start_time = time.time()

import os
import uuid
from swarms import Agent
from swarm_models import OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)


# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)


agent = Agent(
    agent_name=f"{uuid.uuid4().hex}",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path=f"{uuid.uuid4().hex}",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=3000,
    return_step_meta=False,
)

out = agent.run(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria"
)
print(out)

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")
# Execution time: 9.922541856765747 seconds for the whole script
