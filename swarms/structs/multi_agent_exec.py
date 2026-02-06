import asyncio
import concurrent.futures
import os
import sys
from concurrent.futures import (
    ThreadPoolExecutor,
)
from typing import Any, Callable, List, Optional, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.omni_agent_types import AgentType


def run_single_agent(
    agent: AgentType, task: str, *args, **kwargs
) -> Any:
    """
    Run a single agent synchronously with the given task.

    This function provides a synchronous wrapper for executing a single agent
    with a specific task. It passes through any additional arguments and
    keyword arguments to the agent's run method.

    Args:
        agent (AgentType): The agent instance to execute
        task (str): The task string to be executed by the agent
        *args: Variable length argument list passed to agent.run()
        **kwargs: Arbitrary keyword arguments passed to agent.run()

    Returns:
        Any: The result returned by the agent's run method

    Example:
        >>> agent = SomeAgent()
        >>> result = run_single_agent(agent, "Analyze this data")
        >>> print(result)
    """
    return agent.run(task=task, *args, **kwargs)


async def run_agent_async(agent: AgentType, task: str) -> Any:
    """
    Run an agent asynchronously using asyncio event loop.

    This function executes a single agent asynchronously by running it in a
    thread executor to avoid blocking the event loop. It's designed to be
    used within async contexts for concurrent execution.

    Args:
        agent (AgentType): The agent instance to execute asynchronously
        task (str): The task string to be executed by the agent

    Returns:
        Any: The result returned by the agent's run method

    Example:
        >>> async def main():
        ...     agent = SomeAgent()
        ...     result = await run_agent_async(agent, "Process data")
        ...     return result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, run_single_agent, agent, task
    )


async def run_agents_concurrently_async(
    agents: List[AgentType], task: str
) -> List[Any]:
    """
    Run multiple agents concurrently using asyncio gather.

    This function executes multiple agents concurrently using asyncio.gather(),
    which runs all agents in parallel and waits for all to complete. Each agent
    runs the same task asynchronously.

    Args:
        agents (List[AgentType]): List of agent instances to run concurrently
        task (str): The task string to be executed by all agents

    Returns:
        List[Any]: List of results from each agent in the same order as input

    Example:
        >>> async def main():
        ...     agents = [Agent1(), Agent2(), Agent3()]
        ...     results = await run_agents_concurrently_async(agents, "Analyze data")
        ...     for i, result in enumerate(results):
        ...         print(f"Agent {i+1} result: {result}")
    """
    results = await asyncio.gather(
        *(run_agent_async(agent, task) for agent in agents)
    )
    return results


def run_agents_concurrently(
    agents: List["AgentType"],
    task: str,
    img: Optional[str] = None,
    max_workers: Optional[int] = None,
    return_agent_output_dict: bool = False,
) -> Any:
    """
    Execute multiple agents concurrently using a ThreadPoolExecutor.

    This function runs agent tasks in parallel threads, benefitting I/O-bound or mixed-load scenarios.
    Each agent receives the same 'task' (and optional 'img' argument) and runs its .run() method.
    The number of worker threads defaults to 95% of the available CPU cores, unless otherwise specified.

    Args:
        agents (List[AgentType]): List of agent instances to execute concurrently.
        task (str): Task string to pass to all agent run() methods.
        img (Optional[str]): Optional image data to pass to agent run() if supported.
        max_workers (Optional[int]): Maximum threads for the executor (default: 95% of CPU cores).
        return_agent_output_dict (bool): If True, returns a dict mapping agent names to outputs.
                                         Otherwise returns a list of results in completion order.

    Returns:
        List[Any] or Dict[str, Any]: List of results from each agent's run() method in completion order,
                                     or a dict of agent names to results (preserving agent order)
                                     if return_agent_output_dict is True.
                                     If an agent fails, the corresponding result is the Exception.

    Notes:
        - ThreadPoolExecutor is used for efficient, parallel execution.
        - By default, utilizes nearly all available CPU cores for optimal performance.
        - Any Exception during agent execution is caught and included in the results.
        - If return_agent_output_dict is True, the results dict preserves agent input order.
        - Otherwise, the results list is in order of completion (not input order).

    Example:
        >>> agents = [Agent1(), Agent2()]
        >>> # As list
        >>> results = run_agents_concurrently(agents, task="Summarize", img=None)
        >>> # As dict
        >>> results_dict = run_agents_concurrently(
        ...    agents, task="Summarize", return_agent_output_dict=True)
        >>> for name, val in results_dict.items():
        ...     print(f"Result from {name}: {val}")
    """
    try:
        if max_workers is None:
            num_cores = os.cpu_count()
            max_workers = int(num_cores * 0.95) if num_cores else 1

        futures = []
        agent_id_map = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for agent in agents:
                agent_kwargs = {}
                if task is not None:
                    agent_kwargs["task"] = task
                if img is not None:
                    agent_kwargs["img"] = img
                future = executor.submit(agent.run, **agent_kwargs)
                futures.append(future)
                agent_id_map[future] = agent

            if return_agent_output_dict:
                # Use agent name as key, preserve input order
                output_dict = {}
                for agent, future in zip(agents, futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        result = e
                    # Prefer .agent_name or .name, fallback to str(agent)
                    name = (
                        getattr(agent, "agent_name", None)
                        or getattr(agent, "name", None)
                        or str(agent)
                    )
                    output_dict[name] = result
                return output_dict
            else:
                results = []
                for future in concurrent.futures.as_completed(
                    futures
                ):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append(e)
                return results

    except Exception as e:
        logger.error(
            f"Error running_agents_concurrently: {e} Traceback: {e.__traceback__}"
        )
        raise e


def run_agents_concurrently_multiprocess(
    agents: List[Agent], task: str, batch_size: int = os.cpu_count()
) -> List[Any]:
    """
    Run multiple agents concurrently in batches using asyncio for optimized performance.

    This function processes agents in batches to avoid overwhelming system resources
    while still achieving high concurrency. It uses asyncio internally to manage
    the concurrent execution of agent batches.

    Args:
        agents (List[Agent]): List of Agent instances to run concurrently
        task (str): The task string to be executed by all agents
        batch_size (int, optional): Number of agents to run in parallel in each batch.
                                   Defaults to the number of CPU cores for optimal resource usage

    Returns:
        List[Any]: List of results from each agent, maintaining the order of input agents

    Note:
        - Processes agents in batches to prevent resource exhaustion
        - Uses asyncio for efficient concurrent execution within batches
        - Results are returned in the same order as input agents

    Example:
        >>> agents = [Agent1(), Agent2(), Agent3(), Agent4(), Agent5()]
        >>> results = run_agents_concurrently_multiprocess(agents, "Analyze data", batch_size=2)
        >>> print(f"Processed {len(results)} agents")
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
    Run multiple agents with different tasks concurrently using ThreadPoolExecutor.

    This function pairs each agent with a specific task and executes them concurrently.
    It's designed for scenarios where different agents need to work on different tasks
    simultaneously, creating a grid-like execution pattern.

    Args:
        agents (List[AgentType]): List of agent instances to execute
        tasks (List[str]): List of task strings, one for each agent. Must match the number of agents
        max_workers (int, optional): Maximum number of threads to use.
                                   Defaults to 90% of available CPU cores for optimal performance

    Returns:
        List[Any]: List of results from each agent in the same order as input agents.
                  If an agent fails, the exception is included in the results.

    Raises:
        ValueError: If the number of agents doesn't match the number of tasks

    Note:
        - Uses 90% of CPU cores by default for optimal resource utilization
        - Results maintain the same order as input agents
        - Handles exceptions gracefully by including them in results

    Example:
        >>> agents = [Agent1(), Agent2(), Agent3()]
        >>> tasks = ["Task A", "Task B", "Task C"]
        >>> results = batched_grid_agent_execution(agents, tasks)
        >>> for i, result in enumerate(results):
        ...     print(f"Agent {i+1} with {tasks[i]}: {result}")
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

    This function executes each agent on its corresponding task, processing the agent-task pairs
    in batches for efficient resource utilization. It's designed for scenarios where you have
    a large number of agent-task pairs that need to be processed efficiently.

    Args:
        agent_task_pairs (List[tuple[AgentType, str]]): List of (agent, task) tuples to execute.
                                                        Each tuple contains an agent instance and its task
        batch_size (int, optional): Number of agent-task pairs to process in parallel in each batch.
                                   Defaults to 10 for balanced resource usage
        max_workers (int, optional): Maximum number of threads to use for each batch.
                                   If None, uses the default from batched_grid_agent_execution

    Returns:
        List[Any]: List of outputs from each agent-task pair, maintaining the same order as input pairs.
                  If an agent fails, the exception is included in the results.

    Note:
        - Processes agent-task pairs in batches to prevent resource exhaustion
        - Results maintain the same order as input pairs
        - Handles exceptions gracefully by including them in results
        - Uses batched_grid_agent_execution internally for each batch

    Example:
        >>> pairs = [(agent1, "Task A"), (agent2, "Task B"), (agent3, "Task C")]
        >>> results = run_agents_with_different_tasks(pairs, batch_size=5)
        >>> for i, result in enumerate(results):
        ...     agent, task = pairs[i]
        ...     print(f"Agent {agent.agent_name} with {task}: {result}")
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
    Run multiple agents concurrently using optimized async performance with uvloop/winloop.

    This function provides high-performance concurrent execution of multiple agents using
    optimized event loop implementations. It automatically selects the best available
    event loop for the platform (uvloop on Unix systems, winloop on Windows).

    Args:
        agents (List[AgentType]): List of agent instances to run concurrently
        task (str): The task string to be executed by all agents
        max_workers (Optional[int]): Maximum number of threads in the executor.
                                   Defaults to 95% of available CPU cores for optimal performance

    Returns:
        List[Any]: List of results from each agent. If an agent fails, the exception
                  is included in the results list instead of the result.

    Raises:
        ImportError: If neither uvloop nor winloop is available (falls back to standard asyncio)
        RuntimeError: If event loop policy cannot be set (falls back to standard asyncio)

    Note:
        - Automatically uses uvloop on Linux/macOS and winloop on Windows
        - Falls back gracefully to standard asyncio if optimized loops are unavailable
        - Uses 95% of CPU cores by default for optimal resource utilization
        - Handles exceptions gracefully by including them in results
        - Results may not be in the same order as input agents due to concurrent execution

    Example:
        >>> agents = [Agent1(), Agent2(), Agent3()]
        >>> results = run_agents_concurrently_uvloop(agents, "Process data")
        >>> for i, result in enumerate(results):
        ...     if isinstance(result, Exception):
        ...         print(f"Agent {i+1} failed: {result}")
        ...     else:
        ...         print(f"Agent {i+1} result: {result}")
    """
    # Platform-specific event loop policy setup (use stdlib asyncio policies)
    try:
        if sys.platform in ("win32", "cygwin"):
            asyncio.set_event_loop_policy(
                asyncio.WindowsProactorEventLoopPolicy()
            )
            logger.info(
                "Using stdlib WindowsProactorEventLoopPolicy for Windows"
            )
        else:
            asyncio.set_event_loop_policy(
                asyncio.DefaultEventLoopPolicy()
            )
            logger.info(
                "Using stdlib DefaultEventLoopPolicy for Unix-like systems"
            )
    except RuntimeError as e:
        logger.warning(
            f"Could not set asyncio policy: {e}. Continuing with existing policy."
        )

    if max_workers is None:
        # Use 95% of available CPU cores for optimal performance
        num_cores = os.cpu_count()
        max_workers = int(num_cores * 0.95) if num_cores else 1

    logger.info(
        f"Running {len(agents)} agents concurrently with stdlib asyncio (max_workers: {max_workers})"
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
    Run multiple agents with different tasks concurrently using optimized async performance.

    This function pairs each agent with a specific task and runs them concurrently using
    optimized event loop implementations (uvloop on Unix systems, winloop on Windows).
    It's designed for high-performance scenarios where different agents need to work
    on different tasks simultaneously.

    Args:
        agents (List[AgentType]): List of agent instances to run
        tasks (List[str]): List of task strings, one for each agent. Must match the number of agents
        max_workers (Optional[int]): Maximum number of threads in the executor.
                                   Defaults to 95% of available CPU cores for optimal performance

    Returns:
        List[Any]: List of results from each agent in the same order as input agents.
                  If an agent fails, the exception is included in the results.

    Raises:
        ValueError: If the number of agents doesn't match the number of tasks

    Note:
        - Automatically uses uvloop on Linux/macOS and winloop on Windows
        - Falls back gracefully to standard asyncio if optimized loops are unavailable
        - Uses 95% of CPU cores by default for optimal resource utilization
        - Results maintain the same order as input agents
        - Handles exceptions gracefully by including them in results

    Example:
        >>> agents = [Agent1(), Agent2(), Agent3()]
        >>> tasks = ["Task A", "Task B", "Task C"]
        >>> results = run_agents_with_tasks_uvloop(agents, tasks)
        >>> for i, result in enumerate(results):
        ...     if isinstance(result, Exception):
        ...         print(f"Agent {i+1} with {tasks[i]} failed: {result}")
        ...     else:
        ...         print(f"Agent {i+1} with {tasks[i]}: {result}")
    """
    if len(agents) != len(tasks):
        raise ValueError(
            f"Number of agents ({len(agents)}) must match number of tasks ({len(tasks)})"
        )

    # Platform-specific event loop policy setup
    if sys.platform in ("win32", "cygwin"):
        # Windows: Try to use winloop
        try:
            import winloop

            asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
            logger.info(
                "Using winloop for enhanced Windows performance"
            )
        except ImportError:
            logger.warning(
                "winloop not available, falling back to standard asyncio. "
                "Install winloop with: pip install winloop"
            )
        except RuntimeError as e:
            logger.warning(
                f"Could not set winloop policy: {e}. Using default asyncio."
            )
    else:
        # Linux/macOS: Try to use uvloop
        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Using uvloop for enhanced Unix performance")
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

    logger.info(
        f"Running {len(agents)} agents with {len(tasks)} tasks using optimized event loop (max_workers: {max_workers})"
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
    Fetch and format information about all available swarms in the system.

    This function provides a comprehensive overview of all swarms currently
    available in the system, including their names, descriptions, agent counts,
    and swarm types. It's useful for debugging, monitoring, and system introspection.

    Args:
        swarms (List[Callable]): List of swarm instances to get information about.
                               Each swarm should have name, description, agents, and swarm_type attributes

    Returns:
        str: A formatted string containing detailed information about all swarms.
             Returns "No swarms currently available in the system." if the list is empty.

    Note:
        - Each swarm is expected to have the following attributes:
          - name: The name of the swarm
          - description: A description of the swarm's purpose
          - agents: A list of agents in the swarm
          - swarm_type: The type/category of the swarm
        - The output is formatted for human readability with clear section headers

    Example:
        >>> swarms = [swarm1, swarm2, swarm3]
        >>> info = get_swarms_info(swarms)
        >>> print(info)
        Available Swarms:

        [Swarm 1]
        Name: Data Processing Swarm
        Description: Handles data analysis tasks
        Length of Agents: 5
        Swarm Type: Analysis
        ...
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
    Fetch and format information about all available agents in the system.

    This function provides a comprehensive overview of all agents currently
    available in the system, including their names, descriptions, roles,
    models, and configuration details. It's useful for debugging, monitoring,
    and system introspection.

    Args:
        agents (List[Union[Agent, Callable]]): List of agent instances to get information about.
                                             Each agent should have agent_name, agent_description,
                                             role, model_name, and max_loops attributes
        team_name (str, optional): Optional team name to include in the output header.
                                 If None, uses a generic header

    Returns:
        str: A formatted string containing detailed information about all agents.
             Returns "No agents currently available in the system." if the list is empty.

    Note:
        - Each agent is expected to have the following attributes:
          - agent_name: The name of the agent
          - agent_description: A description of the agent's purpose
          - role: The role or function of the agent
          - model_name: The AI model used by the agent
          - max_loops: The maximum number of loops the agent can execute
        - The output is formatted for human readability with clear section headers
        - Team name is included in the header if provided

    Example:
        >>> agents = [agent1, agent2, agent3]
        >>> info = get_agents_info(agents, team_name="Data Team")
        >>> print(info)
        Available Agents for Team: Data Team

        [Agent 1]
        Name: Data Analyzer
        Description: Analyzes data patterns
        Role: Analyst
        Model: gpt-4
        Max Loops: 10
        ...
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
