import os

from swarms import Agent, OpenAIChat, one_to_one

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize the agents
financial_agent1 = Agent(
    agent_name="Financial-Analysis-Agent1",
    # system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent1.json",
    user_name="swarms_corp",
    context_length=200000,
)

financial_agent2 = Agent(
    agent_name="Financial-Analysis-Agent2",
    # system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    streaming_on=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent2.json",
    user_name="swarms_corp",
    context_length=200000,
)

# Agents
task = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria, Discuss with eachother to get the best answer."

# Run the agents in a circular swarm
response = one_to_one(
    sender=financial_agent1,
    receiver=financial_agent2,
    task=task,
    max_loops=1,
)
print(response)
