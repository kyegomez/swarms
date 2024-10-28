import asyncio
from typing import List, Any
from swarms import Agent
from multiprocessing import cpu_count
from swarms.utils.calculate_func_metrics import profile_func

# Use uvloop for faster asyncio event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def run_single_agent(agent: Agent, task: str) -> Any:
    """
    Run a single agent on the given task.

    Args:
        agent (Agent): The agent to run.
        task (str): The task for the agent to perform.

    Returns:
        Any: The result of the agent's execution.
    """
    return agent.run(task)


async def run_agent_async(agent: Agent, task: str) -> Any:
    """
    Asynchronous wrapper for agent tasks.

    Args:
        agent (Agent): The agent to run asynchronously.
        task (str): The task for the agent to perform.

    Returns:
        Any: The result of the agent's execution.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, run_single_agent, agent, task
    )


async def run_agents_concurrently_async(
    agents: List[Agent], task: str
) -> List[Any]:
    """
    Run multiple agents concurrently on the same task with optimized performance.

    Args:
        agents (List[Agent]): List of Agent instances to run concurrently.
        task (str): The task string to execute by all agents.

    Returns:
        List[Any]: A list of outputs from each agent.
    """
    results = await asyncio.gather(
        *(run_agent_async(agent, task) for agent in agents)
    )
    return results


@profile_func
def run_agents_concurrently_multiprocess(
    agents: List[Agent], task: str, batch_size: int = cpu_count()
) -> List[Any]:
    """
    Manage and run multiple agents concurrently in batches, with optimized performance.

    Args:
        agents (List[Agent]): List of Agent instances to run concurrently.
        task (str): The task string to execute by all agents.
        batch_size (int, optional): Number of agents to run in parallel in each batch.
                                    Defaults to the number of CPU cores.

    Returns:
        List[Any]: A list of outputs from each agent.
    """
    results = []
    loop = asyncio.get_event_loop()

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
