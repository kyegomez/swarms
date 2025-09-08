import asyncio
import concurrent.futures
import os
from concurrent.futures import (
    ThreadPoolExecutor,
)
from typing import Any, Callable, List, Optional, Union

import uvloop
from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.omni_agent_types import AgentType


def run_single_agent(
    agent: AgentType, task: str, *args, **kwargs
) -> Any:
    """Run a single agent synchronously"""
    return agent.run(task=task, *args, **kwargs)


async def run_agent_async(agent: AgentType, task: str) -> Any:
    """
    Run an agent asynchronously.

    Args:
        agent: Agent instance to run
        task: Task string to execute

    Returns:
        Agent execution result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, run_single_agent, agent, task
    )


async def run_agents_concurrently_async(
    agents: List[AgentType], task: str
) -> List[Any]:
    """
    Run multiple agents concurrently using asyncio.

    Args:
        agents: List of Agent instances to run concurrently
        task: Task string to execute

    Returns:
        List of outputs from each agent
    """
    results = await asyncio.gather(
        *(run_agent_async(agent, task) for agent in agents)
    )
    return results


def run_agents_concurrently(
    agents: List[AgentType],
    task: str,
    max_workers: Optional[int] = None,
) -> List[Any]:
    """
    Optimized concurrent agent runner using ThreadPoolExecutor.

    Args:
        agents: List of Agent instances to run concurrently
        task: Task string to execute
        max_workers: Maximum number of threads in the executor (defaults to 95% of CPU cores)

    Returns:
        List of outputs from each agent
    """
    if max_workers is None:
        # 95% of the available CPU cores
        num_cores = os.cpu_count()
        max_workers = int(num_cores * 0.95) if num_cores else 1

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        # Submit all tasks and get futures
        futures = [
            executor.submit(agent.run, task) for agent in agents
        ]

        # Wait for all futures to complete and get results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Append the error if an agent fails
                results.append(e)

    return results


def run_agents_concurrently_multiprocess(
    agents: List[Agent], task: str, batch_size: int = os.cpu_count()
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


def batched_grid_agent_execution(
    agents: List["AgentType"],
    tasks: List[str],
    max_workers: int = None,
) -> List[Any]:
    """
    Run multiple agents with different tasks concurrently.

    Args:
        agents (List[AgentType]): List of agent instances.
        tasks (List[str]): List of tasks, one for each agent.
        max_workers (int, optional): Maximum number of threads to use. Defaults to 90% of CPU cores.

    Returns:
        List[Any]: List of results from each agent.
    """
    logger.info(
        f"Batch Grid Execution with {len(agents)} agents and number of tasks: {len(tasks)}"
    )

    if len(agents) != len(tasks):
        raise ValueError(
            "The number of agents must match the number of tasks."
        )

    # 90% of the available CPU cores
    max_workers = max_workers or int(os.cpu_count() * 0.9)

    results = [None] * len(agents)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_index = {
            executor.submit(run_single_agent, agent, task): idx
            for idx, (agent, task) in enumerate(zip(agents, tasks))
        }
        for future in concurrent.futures.as_completed(
            future_to_index
        ):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e

    return results


def run_agents_with_different_tasks(
    agent_task_pairs: List[tuple["AgentType", str]],
    batch_size: int = 10,
    max_workers: int = None,
) -> List[Any]:
    """
    Run multiple agents with different tasks concurrently, processing them in batches.

    This function executes each agent on its corresponding task, processing the agent-task pairs in batches
    of size `batch_size` for efficient resource utilization.

    Args:
        agent_task_pairs: List of (agent, task) tuples.
        batch_size: Number of agents to run in parallel in each batch.
        max_workers: Maximum number of threads.

    Returns:
        List of outputs from each agent, in the same order as the input pairs.
    """
    if not agent_task_pairs:
        return []

    results = []
    total_pairs = len(agent_task_pairs)
    for i in range(0, total_pairs, batch_size):
        batch = agent_task_pairs[i : i + batch_size]
        agents, tasks = zip(*batch)
        batch_results = batched_grid_agent_execution(
            list(agents), list(tasks), max_workers=max_workers
        )
        results.extend(batch_results)
    return results


def run_agents_concurrently_uvloop(
    agents: List[AgentType],
    task: str,
    max_workers: Optional[int] = None,
) -> List[Any]:
    """
    Run multiple agents concurrently using uvloop for optimized async performance.

    uvloop is a fast, drop-in replacement for asyncio's event loop, implemented in Cython.
    It's designed to be significantly faster than the standard asyncio event loop,
    especially beneficial for I/O-bound tasks and concurrent operations.

    Args:
        agents: List of Agent instances to run concurrently
        task: Task string to execute by all agents
        max_workers: Maximum number of threads in the executor (defaults to 95% of CPU cores)

    Returns:
        List of outputs from each agent

    Raises:
        ImportError: If uvloop is not installed
        RuntimeError: If uvloop cannot be set as the event loop policy
    """
    try:
        # Set uvloop as the default event loop policy for better performance
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        logger.warning(
            "uvloop not available, falling back to standard asyncio. "
            "Install uvloop with: pip install uvloop"
        )
    except RuntimeError as e:
        logger.warning(
            f"Could not set uvloop policy: {e}. Using default asyncio."
        )

    if max_workers is None:
        # Use 95% of available CPU cores for optimal performance
        num_cores = os.cpu_count()
        max_workers = int(num_cores * 0.95) if num_cores else 1

    logger.info(
        f"Running {len(agents)} agents concurrently with uvloop (max_workers: {max_workers})"
    )

    async def run_agents_async():
        """Inner async function to handle the concurrent execution."""
        results = []

        def run_agent_sync(agent: AgentType) -> Any:
            """Synchronous wrapper for agent execution."""
            return agent.run(task=task)

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for all agents
            tasks = [
                loop.run_in_executor(executor, run_agent_sync, agent)
                for agent in agents
            ]

            # Wait for all tasks to complete and collect results
            completed_tasks = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            # Handle results and exceptions
            for i, result in enumerate(completed_tasks):
                if isinstance(result, Exception):
                    logger.error(
                        f"Agent {i+1} failed with error: {result}"
                    )
                    results.append(result)
                else:
                    results.append(result)

        return results

    # Run the async function
    try:
        return asyncio.run(run_agents_async())
    except RuntimeError as e:
        if "already running" in str(e).lower():
            # Handle case where event loop is already running
            logger.warning(
                "Event loop already running, using get_event_loop()"
            )
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(run_agents_async())
        else:
            raise


def run_agents_with_tasks_uvloop(
    agents: List[AgentType],
    tasks: List[str],
    max_workers: Optional[int] = None,
) -> List[Any]:
    """
    Run multiple agents with different tasks concurrently using uvloop.

    This function pairs each agent with a specific task and runs them concurrently
    using uvloop for optimized performance.

    Args:
        agents: List of Agent instances to run
        tasks: List of task strings (must match number of agents)
        max_workers: Maximum number of threads (defaults to 95% of CPU cores)

    Returns:
        List of outputs from each agent

    Raises:
        ValueError: If number of agents doesn't match number of tasks
    """
    if len(agents) != len(tasks):
        raise ValueError(
            f"Number of agents ({len(agents)}) must match number of tasks ({len(tasks)})"
        )

    try:
        # Set uvloop as the default event loop policy
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        logger.warning(
            "uvloop not available, falling back to standard asyncio. "
            "Install uvloop with: pip install uvloop"
        )
    except RuntimeError as e:
        logger.warning(
            f"Could not set uvloop policy: {e}. Using default asyncio."
        )

    if max_workers is None:
        num_cores = os.cpu_count()
        max_workers = int(num_cores * 0.95) if num_cores else 1

    logger.inufo(
        f"Running {len(agents)} agents with {len(tasks)} tasks using uvloop (max_workers: {max_workers})"
    )

    async def run_agents_with_tasks_async():
        """Inner async function to handle concurrent execution with different tasks."""
        results = []

        def run_agent_task_sync(agent: AgentType, task: str) -> Any:
            """Synchronous wrapper for agent execution with specific task."""
            return agent.run(task=task)

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks for agent-task pairs
            tasks_async = [
                loop.run_in_executor(
                    executor, run_agent_task_sync, agent, task
                )
                for agent, task in zip(agents, tasks)
            ]

            # Wait for all tasks to complete
            completed_tasks = await asyncio.gather(
                *tasks_async, return_exceptions=True
            )

            # Handle results and exceptions
            for i, result in enumerate(completed_tasks):
                if isinstance(result, Exception):
                    logger.error(
                        f"Agent {i+1} (task: {tasks[i][:50]}...) failed with error: {result}"
                    )
                    results.append(result)
                else:
                    results.append(result)

        return results

    # Run the async function
    try:
        return asyncio.run(run_agents_with_tasks_async())
    except RuntimeError as e:
        if "already running" in str(e).lower():
            logger.warning(
                "Event loop already running, using get_event_loop()"
            )
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                run_agents_with_tasks_async()
            )
        else:
            raise


def get_swarms_info(swarms: List[Callable]) -> str:
    """
    Fetches and formats information about all available swarms in the system.

    Returns:
        str: A formatted string containing names and descriptions of all swarms.
    """
    if not swarms:
        return "No swarms currently available in the system."

    swarm_info = [
        "Available Swarms:",
        "",
    ]  # Empty string for line spacing

    for idx, swarm in enumerate(swarms, 1):
        swarm_info.extend(
            [
                f"[Swarm {idx}]",
                f"Name: {swarm.name}",
                f"Description: {swarm.description}",
                f"Length of Agents: {len(swarm.agents)}",
                f"Swarm Type: {swarm.swarm_type}",
                "",  # Empty string for line spacing between swarms
            ]
        )

    return "\n".join(swarm_info).strip()


def get_agents_info(
    agents: List[Union[Agent, Callable]], team_name: str = None
) -> str:
    """
    Fetches and formats information about all available agents in the system.

    Returns:
        str: A formatted string containing names and descriptions of all swarms.
    """
    if not agents:
        return "No agents currently available in the system."

    agent_info = [
        f"Available Agents for Team: {team_name}",
        "",
    ]  # Empty string for line spacing

    for idx, agent in enumerate(agents, 1):
        agent_info.extend(
            [
                "\n",
                f"[Agent {idx}]",
                f"Name: {agent.agent_name}",
                f"Description: {agent.agent_description}",
                f"Role: {agent.role}",
                f"Model: {agent.model_name}",
                f"Max Loops: {agent.max_loops}",
                "\n",
            ]
        )

    return "\n".join(agent_info).strip()
