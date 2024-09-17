import os

# from swarms.structs. import OpenAIChat
import asyncio
from swarms.utils.calculate_func_metrics import profile_func


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
def run_agents_concurrently(agents, task: str, batch_size: int = 5):
    """
    Manage and run multiple agents concurrently in batches, with optimized performance.

    :param agents: List of Agent instances to run concurrently.
    :param task: The task string to execute by all agents.
    :param batch_size: Number of agents to run in parallel in each batch.
    :return: A list of outputs from each agent.
    """

    results = []
    loop = asyncio.get_event_loop()

    batch_size = (
        os.cpu_count() if batch_size > os.cpu_count() else batch_size
    )

    # Process agents in batches to avoid overwhelming system resources
    for i in range(0, len(agents), batch_size):
        batch = agents[i : i + batch_size]
        batch_results = loop.run_until_complete(
            run_agents_concurrently_async(batch, task)
        )
        results.extend(batch_results)

    return results


# # Example usage:
# # Initialize your agents with the same model to avoid re-creating it
# agents = [
#     Agent(
#         agent_name=f"Financial-Analysis-Agent_parallel_swarm{i}",
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
# outputs = run_agents_concurrently(agents, task)

# for i, output in enumerate(outputs):
#     print(f"Output from agent {i+1}:\n{output}")

# # Output from agent 2:
# # execution_time=12.89196228981018 memory_usage=-294.9375 cpu_usage=-10.3 io_operations=23309 function_calls=1
# # execution_time=11.810921907424927 memory_usage=-242.734375 cpu_usage=-26.4 io_operations=10752 function_calls=1

# # Parallel
# # execution_time=18.79391312599182 memory_usage=-342.9375 cpu_usage=-2.5 io_operations=59518 function_calls=1

# # # Multiprocess
# # 2024-08-22T14:49:33.986491-0400 Function metrics: {
# #     "execution_time": 24.783875942230225,
# #     "memory_usage": -286.734375,
# #     "cpu_usage": -24.6,
# #     "io_operations": 17961,
# #     "function_calls": 1
# # }


# # Latest
# # Analysis-Agent_parallel_swarm4_state.json
# # 2024-08-22T15:43:11.800970-0400 Function metrics: {
# #     "execution_time": 11.062030792236328,
# #     "memory_usage": -249.5625,
# #     "cpu_usage": -15.700000000000003,
# #     "io_operations": 13439,
# #     "function_calls": 1
# # }
