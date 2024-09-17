import os

from swarms import Agent
from swarm_models import OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

# Example usage:
api_key = os.getenv("OPENAI_API_KEY")

# Model
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)


# Initialize your agents (assuming the Agent class and model are already defined)
agents = [
    Agent(
        agent_name=f"Financial-Analysis-Agent-spreesheet-swarm:{i}",
        system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="finance_agent.json",
        user_name="swarms_corp",
        retry_attempts=1,
    )
    for i in range(10)
]

# Create a Swarm with the list of agents
swarm = SpreadSheetSwarm(
    name="Finance-Spreadsheet-Swarm",
    description="A swarm that processes tasks from a queue using multiple agents on different threads.",
    agents=agents,
    autosave_on=True,
    save_file_path="financial_spreed_sheet_swarm_demo.csv",
    run_all_agents=False,
    max_loops=1,
)

# Run the swarm
swarm.run(
    task="Analyze the states with the least taxes for LLCs. Provide an overview of all tax rates and add them with a comprehensive analysis"
)
