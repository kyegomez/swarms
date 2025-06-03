from swarms.structs.agent import Agent
from typing import List
from swarms.utils.formatter import formatter


def batch_agent_execution(
    agents: List[Agent],
    tasks: List[str],
):
    """
    Execute a batch of agents on a list of tasks concurrently.

    Args:
        agents (List[Agent]): List of agents to execute
        tasks (list[str]): List of tasks to execute

    Returns:
        List[str]: List of results from each agent execution

    Raises:
        ValueError: If number of agents doesn't match number of tasks
    """
    if len(agents) != len(tasks):
        raise ValueError(
            "Number of agents must match number of tasks"
        )

    import concurrent.futures
    import multiprocessing

    results = []

    # Calculate max workers as 90% of available CPU cores
    max_workers = max(1, int(multiprocessing.cpu_count() * 0.9))

    formatter.print_panel(
        f"Executing {len(agents)} agents on {len(tasks)} tasks using {max_workers} workers"
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        # Submit all tasks to the executor
        future_to_task = {
            executor.submit(agent.run, task): (agent, task)
            for agent, task in zip(agents, tasks)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            agent, task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(
                    f"Task failed for agent {agent.agent_name}: {str(e)}"
                )
                results.append(None)

        # Wait for all futures to complete before returning
        concurrent.futures.wait(future_to_task.keys())

    return results
