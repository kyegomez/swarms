import os

from swarms.structs.queue_swarm import TaskQueueSwarm
from swarms import Agent, OpenAIChat
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)
from swarms.utils.calculate_func_metrics import profile_func

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
    for i in range(2)
]
# Create a Swarm with the list of agents
swarm = TaskQueueSwarm(
    agents=agents,
    return_metadata_on=True,
    autosave_on=True,
    save_file_path="swarm_run_metadata.json",
)


@profile_func
def execute_task_queue_swarm():
    # Add tasks to the swarm
    swarm.add_task(
        "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"
    )
    swarm.add_task(
        "Analyze the financial risks of investing in tech stocks."
    )

    # Keep adding tasks as needed...
    # swarm.add_task("...")

    # Run the swarm and get the output
    out = swarm.run()

    # Print the output
    print(out)

    # Export the swarm metadata
    swarm.export_metadata()


execute_task_queue_swarm()


# 2024-08-27T14:06:16.278473-0400 Function metrics: {
#     "execution_time": 10.653800249099731,
#     "memory_usage": -386.15625,
#     "cpu_usage": 3.6000000000000014,
#     "io_operations": 13566,
#     "function_calls": 1
# }

# 2024-08-27T14:06:32.640856-0400 Function metrics: {
#     "execution_time": 8.788740873336792,
#     "memory_usage": -396.078125,
#     "cpu_usage": -4.399999999999999,
#     "io_operations": 4014,
#     "function_calls": 1
# }
