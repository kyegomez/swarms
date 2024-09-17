import os
import asyncio
from swarms import Agent
from swarm_models import OpenAIChat
import uvloop
from multiprocessing import cpu_count
from swarms.utils.calculate_func_metrics import profile_func
from typing import List

# Use uvloop for faster asyncio event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class (can be reused)
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)


# Function to run a single agent on the task (synchronous)
def run_single_agent(agent, task):
    return agent.run(task)


# Asynchronous wrapper for agent tasks
async def run_agent_async(agent, task):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, run_single_agent, agent, task
    )


# Asynchronous function to run agents concurrently
async def run_agents_concurrently_async(agents, task: str):
    """
    Run multiple agents concurrently on the same task with optimized performance.

    :param agents: List of Agent instances to run concurrently.
    :param task: The task string to execute by all agents.
    :return: A list of outputs from each agent.
    """

    # Run all agents asynchronously using asyncio.gather
    results = await asyncio.gather(
        *(run_agent_async(agent, task) for agent in agents)
    )
    return results


# Function to manage the overall process and batching
@profile_func
def run_agents_concurrently_multiprocess(
    agents: List[Agent], task: str, batch_size: int = cpu_count()
):
    """
    Manage and run multiple agents concurrently in batches, with optimized performance.

    :param agents: List of Agent instances to run concurrently.
    :param task: The task string to execute by all agents.
    :param batch_size: Number of agents to run in parallel in each batch.
    :return: A list of outputs from each agent.
    """

    results = []
    loop = asyncio.get_event_loop()
    # batch_size = cpu_count()

    # Process agents in batches to avoid overwhelming system resources
    for i in range(0, len(agents), batch_size):
        batch = agents[i : i + batch_size]
        batch_results = loop.run_until_complete(
            run_agents_concurrently_async(batch, task)
        )
        results.extend(batch_results)

    return results


# # # Example usage:
# # Initialize your agents with the same model to avoid re-creating it
# agents = [
#     Agent(
#         agent_name=f"Financial-Analysis-Agent_new_parallel_swarm_test{i}",
#         system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#         llm=model,
#         max_loops=1,
#         autosave=True,
#         dashboard=False,
#         verbose=False,
#         dynamic_temperature_enabled=False,
#         saved_state_path=f"finance_agent_{i}.json",
#         user_name="swarms_corp",
#         retry_attempts=1,
#         context_length=200000,
#         return_step_meta=False,
#     )
#     for i in range(5)  # Assuming you want 10 agents
# ]

# task = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria"
# outputs = run_agents_concurrently_multiprocess(
#     agents,
#     task,
# )

# for i, output in enumerate(outputs):
#     print(f"Output from agent {i+1}:\n{output}")


# # execution_time=15.958749055862427 memory_usage=-328.046875 cpu_usage=-2.5999999999999943 io_operations=81297 function_calls=1
# # Analysis-Agent_new_parallel_swarm_test1_state.json
# # 2024-08-22T15:42:12.463246-0400 Function metrics: {
# #     "execution_time": 15.958749055862427,
# #     "memory_usage": -328.046875,
# #     "cpu_usage": -2.5999999999999943,
# #     "io_operations": 81297,
#     "function_calls": 1
# }
