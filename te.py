import gevent
from gevent import monkey, pool
import asyncio
from functools import wraps
from typing import Callable, List, Tuple, Union, Optional, Any, Dict
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

# Move monkey patching to the top and be more specific about what we patch
monkey.patch_all(thread=False, select=False, ssl=False)


@dataclass
class TaskMetrics:
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error: Optional[Exception] = None
    retries: int = 0


class TaskExecutionError(Exception):
    """Custom exception for task execution errors"""

    def __init__(self, task_name: str, error: Exception):
        self.task_name = task_name
        self.original_error = error
        super().__init__(
            f"Task {task_name} failed with error: {str(error)}"
        )


@contextmanager
def task_timer(task_name: str):
    """Context manager for timing task execution"""
    start_time = datetime.now()
    try:
        yield
    finally:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.debug(
            f"Task {task_name} completed in {duration:.2f} seconds"
        )


def with_retries(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(
                            delay * (attempt + 1)
                        )  # Exponential backoff
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}"
                        )
                    else:
                        logger.error(
                            f"All {max_retries} retries failed for {func.__name__}"
                        )
                        return last_exception  # Return the exception instead of raising it
            return last_exception

        return wrapper

    return decorator


def run_concurrently_greenlets(
    tasks: List[Union[Callable, Tuple[Callable, tuple, dict]]],
    timeout: Optional[float] = None,
    max_concurrency: int = 100,
    max_retries: int = 3,
    task_timeout: Optional[float] = None,
    metrics: Optional[Dict[str, TaskMetrics]] = None,
) -> List[Any]:
    """
    Execute multiple tasks concurrently using gevent greenlets.

    Args:
        tasks: List of tasks to execute. Each task can be a callable or a tuple of (callable, args, kwargs)
        timeout: Global timeout for all tasks in seconds
        max_concurrency: Maximum number of concurrent tasks
        max_retries: Maximum number of retries per task
        task_timeout: Individual task timeout in seconds
        metrics: Optional dictionary to store task execution metrics

    Returns:
        List of results from all tasks. Failed tasks will return their exception.
    """
    if metrics is None:
        metrics = {}

    pool_obj = pool.Pool(max_concurrency)
    jobs = []
    start_time = datetime.now()

    def wrapper(task_info):
        if isinstance(task_info, tuple):
            fn, args, kwargs = task_info
        else:
            fn, args, kwargs = task_info, (), {}

        task_name = (
            fn.__name__ if hasattr(fn, "__name__") else str(fn)
        )
        metrics[task_name] = TaskMetrics(start_time=datetime.now())

        with task_timer(task_name):
            try:
                if asyncio.iscoroutinefunction(fn):
                    # Handle async functions
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        if task_timeout:
                            result = asyncio.wait_for(
                                fn(*args, **kwargs),
                                timeout=task_timeout,
                            )
                        else:
                            result = loop.run_until_complete(
                                fn(*args, **kwargs)
                            )
                        metrics[task_name].success = True
                        return result
                    finally:
                        loop.close()
                else:
                    if task_timeout:
                        with gevent.Timeout(
                            task_timeout,
                            TimeoutError(
                                f"Task {task_name} timed out after {task_timeout} seconds"
                            ),
                        ):
                            result = fn(*args, **kwargs)
                    else:
                        result = fn(*args, **kwargs)

                    if isinstance(result, Exception):
                        metrics[task_name].error = result
                        return result

                    metrics[task_name].success = True
                    return result
            except Exception as e:
                metrics[task_name].error = e
                logger.exception(
                    f"Task {task_name} failed with error: {str(e)}"
                )
                return TaskExecutionError(task_name, e)
            finally:
                metrics[task_name].end_time = datetime.now()

    try:
        for task in tasks:
            jobs.append(pool_obj.spawn(wrapper, task))

        gevent.joinall(jobs, timeout=timeout)

        results = []
        for job in jobs:
            if job.ready():
                results.append(job.value)
            else:
                timeout_error = TimeoutError("Task timed out")
                results.append(timeout_error)
                if hasattr(job, "value") and hasattr(
                    job.value, "__name__"
                ):
                    metrics[job.value.__name__].error = timeout_error
                    metrics[job.value.__name__].end_time = (
                        datetime.now()
                    )

        return results
    except Exception:
        logger.exception("Fatal error in task execution")
        raise
    finally:
        # Cleanup
        pool_obj.kill()
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Total execution time: {execution_time:.2f} seconds"
        )

        # Log metrics summary
        success_count = sum(1 for m in metrics.values() if m.success)
        failure_count = len(metrics) - success_count
        logger.info(
            f"Task execution summary: {success_count} succeeded, {failure_count} failed"
        )


# # Example tasks
# @with_retries(max_retries=3)
# def task_1(x: int, y: int):
#     import time

#     time.sleep(1)
#     return f"task 1 done with {x + y}"


# @with_retries(max_retries=3)
# def task_3():
#     import time

#     time.sleep(0.5)
#     return "task 3 done"


# async def async_task(x: int):
#     await asyncio.sleep(1)
#     return f"async task done with {x}"


# if __name__ == "__main__":
#     # Example usage with different types of tasks
#     tasks = [
#         (task_1, (1, 2), {}),  # Function with args
#         (task_3, (), {}),  # Function without args (explicit)
#         (async_task, (42,), {}),  # Async function
#     ]

#     results = run_concurrently_greenlets(
#         tasks, timeout=5, max_concurrency=10, max_retries=3
#     )

#     for i, result in enumerate(results):
#         if isinstance(result, Exception):
#             print(f"Task {i} failed with {result}")
#         else:
#             print(f"Task {i} succeeded with result: {result}")
