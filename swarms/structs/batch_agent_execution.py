import concurrent.futures
import os
import traceback
from typing import Callable, List, Optional, Union

from loguru import logger

from swarms.structs.agent import Agent
from swarms.utils.formatter import formatter


class BatchAgentExecutionError(Exception):
    pass


def batch_agent_execution(
    agents: List[Union[Agent, Callable]],
    tasks: Optional[List[str]] = None,
    imgs: Optional[List[str]] = None,
    max_workers: int = max(1, int(os.cpu_count() * 0.9)),
):
    """
    Concurrently execute agents (or callables) on tasks (with optional images).

    Args:
        agents (List[Agent|Callable]): Agents/callables to run.
        tasks (List[str]): Tasks (one per agent).
        imgs (List[str], optional): Images for agents; None for no images.
        max_workers (int): Thread pool size.

    Returns:
        List[Any]: Results (aligned to agents, None if exception).

    Raises:
        BatchAgentExecutionError: On batch setup/run errors.
        ValueError: On length mismatch between agents, tasks, imgs.
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

        img_list = [
            imgs[index] if imgs is not None else None
            for index in range(len(agents))
        ]

        names = [
            getattr(agent, "agent_name", repr(agent))
            for agent in agents
        ]

        results = [None] * len(agents)

        formatter.print_panel(
            f"Executing {len(agents)} agents on {len(tasks)} tasks using {max_workers} workers"
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_index = {
                executor.submit(
                    getattr(agent, "run", agent), task, img
                ): index
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
