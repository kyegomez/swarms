import concurrent.futures
import os
import traceback
from typing import Any, Callable, List, Union

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
        imgs (list[str]): Optional list of image paths, one per agent/task pair

    Returns:
        List[str]: List of results from each agent execution, in the same
            order as ``agents``/``tasks`` regardless of completion order

    Raises:
        ValueError: If number of agents doesn't match number of tasks or imgs
    """
    try:

        logger.info(
            f"Executing {len(agents)} agents on {len(tasks)} tasks"
        )

        if len(agents) != len(tasks):
            raise ValueError(
                "Number of agents must match number of tasks"
            )

        if imgs is None:
            imgs = [None] * len(agents)
        elif len(imgs) != len(agents):
            raise ValueError(
                "Number of imgs must match number of agents"
            )

        results: List[Any] = [None] * len(agents)

        formatter.print_panel(
            f"Executing {len(agents)} agents on {len(tasks)} tasks using {max_workers} workers"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all tasks to the executor, tracking each future's index
            # so results land back in agents[i]/tasks[i] order regardless of
            # completion order.
            future_to_index = {
                executor.submit(agent.run, task, img): index
                for index, (agent, task, img) in enumerate(
                    zip(agents, tasks, imgs)
                )
            }

            for future in concurrent.futures.as_completed(
                future_to_index
            ):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(
                        f"Task failed for agent {agents[index].agent_name}: {str(e)}"
                    )

        return results
    except Exception as e:
        log = f"Batch agent execution failed Error: {str(e)} Traceback: {traceback.format_exc()}"

        logger.error(log)

        raise BatchAgentExecutionError(log)
