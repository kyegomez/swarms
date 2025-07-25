import asyncio
from swarms import Agent
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.utils.visualizer import (
    SwarmVisualizationRich,
    SwarmMetadata,
)  # Replace with your actual module name

# Create two example agents
agent1 = Agent(
    agent_name="Financial-Analysis-Agent",
    system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
    model_name="gpt-4o-mini",
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
    output_type="string",
    streaming_on=False,
)

# Create a second dummy agent for demonstration
agent2 = Agent(
    agent_name="Stock-Advisor-Agent",
    system_prompt="Provide stock market insights and investment advice.",
    model_name="gpt-4o-mini",
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="stock_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
    output_type="string",
    streaming_on=False,
)

# Create swarm metadata
metadata = SwarmMetadata(
    name="Financial Swarm",
    description="A swarm of agents focused on financial analysis and stock market advice.",
    version="1.0",
    author="Your Name",
    primary_objective="Provide comprehensive financial and investment analysis.",
)

# Instantiate the visualizer with a list of agents
visualizer = SwarmVisualizationRich(
    swarm_metadata=metadata,
    agents=[agent1, agent2],
    update_resources=True,
    refresh_rate=0.1,
)

# Start the visualization
asyncio.run(visualizer.start())
