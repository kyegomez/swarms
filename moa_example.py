import os

from swarm_models import OpenAIChat

from swarms import Agent, MixtureOfAgents

# Example usage:
api_key = os.getenv("OPENAI_API_KEY")

# Create individual agents with the OpenAIChat model
model1 = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)
model2 = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)
model3 = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

agent1 = Agent(
    agent_name="Agent1",
    llm=model1,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="agent1_state.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

agent2 = Agent(
    agent_name="Agent2",
    llm=model2,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="agent2_state.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

agent3 = Agent(
    agent_name="Agent3",
    llm=model3,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="agent3_state.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

aggregator_agent = Agent(
    agent_name="AggregatorAgent",
    llm=model1,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="aggregator_agent_state.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

# Create the Mixture of Agents class
moa = MixtureOfAgents(
    reference_agents=[agent1, agent2, agent3],
    aggregator_agent=aggregator_agent,
    aggregator_system_prompt="""You have been provided with a set of responses from various agents.
    Your task is to synthesize these responses into a single, high-quality response.""",
    layers=3,
)

out = moa.run(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"
)
print(out)
