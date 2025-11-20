import concurrent.futures
import os
import traceback
from typing import Callable, List, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.utils.formatter import formatter


class BatchAgentExecutionError(Exception):
    pass


def batch_agent_execution(
    agents: List[Union[Agent, Callable]],
    tasks: List[str] = None,
    imgs: List[str] = None,
    max_workers: int = max(1, int(os.cpu_count() * 0.9)),
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
    try:

        logger.info(
            f"Executing {len(agents)} agents on {len(tasks)} tasks"
        )

        if len(agents) != len(tasks):
            raise ValueError(
                "Number of agents must match number of tasks"
            )

        results = []

        formatter.print_panel(
            f"Executing {len(agents)} agents on {len(tasks)} tasks using {max_workers} workers"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all tasks to the executor
            future_to_task = {
                executor.submit(agent.run, task, imgs): (
                    agent,
                    task,
                    imgs,
                )
                for agent, task, imgs in zip(agents, tasks, imgs)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(
                future_to_task
            ):
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
    except Exception as e:
        log = f"Batch agent execution failed Error: {str(e)} Traceback: {traceback.format_exc()}"

        logger.error(log)

        raise BatchAgentExecutionError(log)
