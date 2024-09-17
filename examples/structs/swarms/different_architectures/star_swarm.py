import os
from swarms import Agent
from swarm_models import OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.structs.swarming_architectures import (
    star_swarm,
)

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize the agents
financial_agent1 = Agent(
    agent_name="Financial-Analysis-Agent1",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent1.json",
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=200000,
)

financial_agent2 = Agent(
    agent_name="Financial-Analysis-Agent2",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent2.json",
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=200000,
)

financial_agent3 = Agent(
    agent_name="Financial-Analysis-Agent3",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent3.json",
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=200000,
)

financial_agent4 = Agent(
    agent_name="Financial-Analysis-Agent4",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent4.json",
    user_name="swarms_corp",
    retry_attempts=3,
    context_length=200000,
)

# Agents
agents = [
    financial_agent1,
    financial_agent2,
    financial_agent3,
    financial_agent4,
]
task = [
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria"
    "What's the best platform to setup a trust?"
    "How can I get a loan to start a business?",
]

# Run the agents in a circular swarm
response = star_swarm(
    agents=agents,
    tasks=task,
)
print(response)
