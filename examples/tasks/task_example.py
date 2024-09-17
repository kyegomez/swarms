import os
from datetime import datetime, timedelta

from swarms import Agent, OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.structs.task import Task
from swarms.utils.loguru_logger import logger

# Example setup

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent_sas_chicken_eej",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)


# Define a task with a condition and action
def condition_check():
    # Example condition: Check if a certain file exists
    return os.path.exists("finance_agent.json")


def post_execution_action():
    logger.info("Task completed! Post execution action triggered.")


# Schedule the task to run 10 seconds from now
schedule_time = datetime.now() + timedelta(seconds=10)

# Schedule the task for 1 month from now
# schedule_time = datetime.now() + timedelta(days=30)

# Create the task instance
task = Task(
    agent=agent,
    description="How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria",
    condition=condition_check,
    action=post_execution_action,
    schedule_time=schedule_time,
    trigger=None,
)

# Run the task
task.run(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria"
)

# The task will be scheduled to run at the specified time, check the condition,
# execute if the condition is met, and perform the post-execution action.
