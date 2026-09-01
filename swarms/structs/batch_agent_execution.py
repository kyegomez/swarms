import concurrent.futures
import os
import traceback
from typing import Any, Callable, List, Optional, Union

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

    Each agent is paired with the task at the same index: ``agents[i]``
    runs ``tasks[i]`` with ``imgs[i]``.

    Args:
        agents (List[Union[Agent, Callable]]): Agents to execute, one per task.
        tasks (List[str]): Tasks to execute, one per agent.
        imgs (List[str], optional): Image passed to each agent alongside its
            task, one per agent. Defaults to None, meaning no images.
        max_workers (int): Cap on threads used to run the batch.

    Returns:
        List[Any]: One result per agent, in the order the agents were given.
            An agent that raised leaves ``None`` in its own slot.

    Raises:
        BatchAgentExecutionError: Wrapping any failure to set up or run the
            batch, including the length mismatches below.
        ValueError: If the number of agents, tasks or imgs disagree.

    Notes:
        Results are placed by index rather than appended on completion, so
        the returned list is aligned with ``agents`` no matter what order
        the threads finish in. Callers pair results with agents positionally
        and have nothing else to key on.
    """
    try:

        logger.info(
            f"Executing {len(agents)} agents on {len(tasks)} tasks"
        )

        if len(agents) != len(tasks):
            raise ValueError(
                "Number of agents must match number of tasks"
            )

        if imgs is not None and len(imgs) != len(agents):
            raise ValueError(
                "Number of imgs must match number of agents"
            )

        img_list: List[Optional[str]] = [
            imgs[index] if imgs is not None else None
            for index in range(len(agents))
        ]
        names = [
            getattr(agent, "agent_name", repr(agent))
            for agent in agents
        ]
        results: List[Any] = [None] * len(agents)

        formatter.print_panel(
            f"Executing {len(agents)} agents on {len(tasks)} tasks using {max_workers} workers"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_index = {
                executor.submit(agent.run, task, img): index
                for index, (agent, task, img) in enumerate(
                    zip(agents, tasks, img_list)
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
                        f"Task failed for agent {names[index]}: {e}"
                    )

        return results
    except Exception as e:
        log = f"Batch agent execution failed Error: {str(e)} Traceback: {traceback.format_exc()}"

        logger.error(log)

        raise BatchAgentExecutionError(log)
