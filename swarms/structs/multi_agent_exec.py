import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
from dataclasses import dataclass
import threading
from typing import List, Union, Any, Callable
from multiprocessing import cpu_count


from swarms.structs.agent import Agent
from swarms.utils.calculate_func_metrics import profile_func


# Type definitions
AgentType = Union[Agent, Callable]


def run_single_agent(agent: AgentType, task: str) -> Any:
    """Run a single agent synchronously"""
    return agent.run(task)


async def run_agent_async(
    agent: AgentType, task: str, executor: ThreadPoolExecutor
) -> Any:
    """
    Run an agent asynchronously using a thread executor.

    Args:
        agent: Agent instance to run
        task: Task string to execute
        executor: ThreadPoolExecutor instance for handling CPU-bound operations

    Returns:
        Agent execution result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, run_single_agent, agent, task
    )


async def run_agents_concurrently_async(
    agents: List[AgentType], task: str, executor: ThreadPoolExecutor
) -> List[Any]:
    """
    Run multiple agents concurrently using asyncio and thread executor.

    Args:
        agents: List of Agent instances to run concurrently
        task: Task string to execute
        executor: ThreadPoolExecutor for CPU-bound operations

    Returns:
        List of outputs from each agent
    """
    results = await asyncio.gather(
        *(run_agent_async(agent, task, executor) for agent in agents)
    )
    return results


@profile_func
def run_agents_concurrently(
    agents: List[AgentType],
    task: str,
    batch_size: int = None,
    max_workers: int = None,
) -> List[Any]:
    """
    Optimized concurrent agent runner using both uvloop and ThreadPoolExecutor.

    Args:
        agents: List of Agent instances to run concurrently
        task: Task string to execute
        batch_size: Number of agents to run in parallel in each batch (defaults to CPU count)
        max_workers: Maximum number of threads in the executor (defaults to CPU count * 2)

    Returns:
        List of outputs from each agent
    """
    # Optimize defaults based on system resources
    cpu_cores = cpu_count()
    batch_size = batch_size or cpu_cores
    max_workers = max_workers or cpu_cores * 2

    results = []

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Create a shared thread pool executor with optimal worker count
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process agents in batches
        for i in range(0, len(agents), batch_size):
            batch = agents[i : i + batch_size]
            batch_results = loop.run_until_complete(
                run_agents_concurrently_async(batch, task, executor)
            )
            results.extend(batch_results)

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


@profile_func
def run_agents_sequentially(
    agents: List[AgentType], task: str
) -> List[Any]:
    """
    Run multiple agents sequentially for baseline comparison.

    Args:
        agents: List of Agent instances to run
        task: Task string to execute

    Returns:
        List of outputs from each agent
    """
    return [run_single_agent(agent, task) for agent in agents]


@profile_func
def run_agents_with_different_tasks(
    agent_task_pairs: List[tuple[AgentType, str]],
    batch_size: int = None,
    max_workers: int = None,
) -> List[Any]:
    """
    Run multiple agents with different tasks concurrently.

    Args:
        agent_task_pairs: List of (agent, task) tuples
        batch_size: Number of agents to run in parallel
        max_workers: Maximum number of threads

    Returns:
        List of outputs from each agent
    """

    async def run_pair_async(
        pair: tuple[AgentType, str], executor: ThreadPoolExecutor
    ) -> Any:
        agent, task = pair
        return await run_agent_async(agent, task, executor)

    cpu_cores = cpu_count()
    batch_size = batch_size or cpu_cores
    max_workers = max_workers or cpu_cores * 2
    results = []

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(agent_task_pairs), batch_size):
            batch = agent_task_pairs[i : i + batch_size]
            batch_results = loop.run_until_complete(
                asyncio.gather(
                    *(
                        run_pair_async(pair, executor)
                        for pair in batch
                    )
                )
            )
            results.extend(batch_results)

    return results


async def run_agent_with_timeout(
    agent: AgentType,
    task: str,
    timeout: float,
    executor: ThreadPoolExecutor,
) -> Any:
    """
    Run an agent with a timeout limit.

    Args:
        agent: Agent instance to run
        task: Task string to execute
        timeout: Timeout in seconds
        executor: ThreadPoolExecutor instance

    Returns:
        Agent execution result or None if timeout occurs
    """
    try:
        return await asyncio.wait_for(
            run_agent_async(agent, task, executor), timeout=timeout
        )
    except asyncio.TimeoutError:
        return None


@profile_func
def run_agents_with_timeout(
    agents: List[AgentType],
    task: str,
    timeout: float,
    batch_size: int = None,
    max_workers: int = None,
) -> List[Any]:
    """
    Run multiple agents concurrently with a timeout for each agent.

    Args:
        agents: List of Agent instances
        task: Task string to execute
        timeout: Timeout in seconds for each agent
        batch_size: Number of agents to run in parallel
        max_workers: Maximum number of threads

    Returns:
        List of outputs (None for timed out agents)
    """
    cpu_cores = cpu_count()
    batch_size = batch_size or cpu_cores
    max_workers = max_workers or cpu_cores * 2
    results = []

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(agents), batch_size):
            batch = agents[i : i + batch_size]
            batch_results = loop.run_until_complete(
                asyncio.gather(
                    *(
                        run_agent_with_timeout(
                            agent, task, timeout, executor
                        )
                        for agent in batch
                    )
                )
            )
            results.extend(batch_results)

    return results


@dataclass
class ResourceMetrics:
    cpu_percent: float
    memory_percent: float
    active_threads: int


def get_system_metrics() -> ResourceMetrics:
    """Get current system resource usage"""
    return ResourceMetrics(
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        active_threads=threading.active_count(),
    )


@profile_func
def run_agents_with_resource_monitoring(
    agents: List[AgentType],
    task: str,
    cpu_threshold: float = 90.0,
    memory_threshold: float = 90.0,
    check_interval: float = 1.0,
) -> List[Any]:
    """
    Run agents with system resource monitoring and adaptive batch sizing.

    Args:
        agents: List of Agent instances
        task: Task string to execute
        cpu_threshold: Max CPU usage percentage
        memory_threshold: Max memory usage percentage
        check_interval: Resource check interval in seconds

    Returns:
        List of outputs from each agent
    """

    async def monitor_resources():
        while True:
            metrics = get_system_metrics()
            if (
                metrics.cpu_percent > cpu_threshold
                or metrics.memory_percent > memory_threshold
            ):
                # Reduce batch size or pause execution
                pass
            await asyncio.sleep(check_interval)

    # Implementation details...


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
