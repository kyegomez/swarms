import os

from swarms.structs.queue_swarm import TaskQueueSwarm
from swarms import Agent, OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Example usage:
api_key = os.getenv("OPENAI_API_KEY")

# Model
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)


# Initialize your agents (assuming the Agent class and model are already defined)
agents = [
    Agent(
        agent_name=f"Financial-Analysis-Agent-Task-Queue-swarm-{i}",
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
    for i in range(10)
]
# Create a Swarm with the list of agents
swarm = TaskQueueSwarm(
    agents=agents,
    return_metadata_on=True,
    autosave_on=True,
    save_file_path="swarm_run_metadata.json",
)

# Add tasks to the swarm
swarm.add_task(
    "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"
)
swarm.add_task("Analyze the financial risks of investing in tech stocks.")

# Keep adding tasks as needed...
# swarm.add_task("...")

# Run the swarm and get the output
out = swarm.run()

# Print the output
print(out)

# Export the swarm metadata
swarm.export_metadata()
