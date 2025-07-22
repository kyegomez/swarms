import os
import asyncio
import concurrent.futures
import inspect
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from functools import wraps
from typing import (
    Any,
    Callable,
    List,
    Optional,
    TypeVar,
    Generic,
)
from dataclasses import dataclass
from enum import Enum

from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger("concurrent_wrapper")

T = TypeVar("T")
R = TypeVar("R")


# Global function for process pool execution (must be picklable)
def _execute_task_in_process(task_data):
    """
    Execute a task in a separate process.
    This function must be at module level to be picklable.
    """
    (
        func,
        task_args,
        task_kwargs,
        task_id,
        max_retries,
        retry_on_failure,
        retry_delay,
        return_exceptions,
    ) = task_data

    start_time = time.time()

    for attempt in range(max_retries + 1):
        try:
            result = func(*task_args, **task_kwargs)
            execution_time = time.time() - start_time
            return ConcurrentResult(
                value=result,
                execution_time=execution_time,
                worker_id=task_id,
            )
        except Exception as e:
            if attempt == max_retries or not retry_on_failure:
                execution_time = time.time() - start_time
                if return_exceptions:
                    return ConcurrentResult(
                        exception=e,
                        execution_time=execution_time,
                        worker_id=task_id,
                    )
                else:
                    raise
            else:
                time.sleep(retry_delay * (2**attempt))

    # This should never be reached, but just in case
    return ConcurrentResult(
        exception=Exception("Max retries exceeded")
    )


class ExecutorType(Enum):
    """Enum for different types of executors."""

    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"


@dataclass
class ConcurrentConfig:
    """Configuration for concurrent execution."""

    name: Optional[str] = None
    description: Optional[str] = None
    max_workers: int = 4
    timeout: Optional[float] = None
    executor_type: ExecutorType = ExecutorType.THREAD
    return_exceptions: bool = False
    chunk_size: Optional[int] = None
    ordered: bool = True
    retry_on_failure: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0


class ConcurrentResult(Generic[T]):
    """Result wrapper for concurrent execution."""

    def __init__(
        self,
        value: T = None,
        exception: Exception = None,
        execution_time: float = 0.0,
        worker_id: Optional[int] = None,
    ):
        self.value = value
        self.exception = exception
        self.execution_time = execution_time
        self.worker_id = worker_id
        self.success = exception is None

    def __repr__(self):
        if self.success:
            return f"ConcurrentResult(value={self.value}, time={self.execution_time:.3f}s)"
        else:
            return f"ConcurrentResult(exception={type(self.exception).__name__}: {self.exception})"


def concurrent(
    name: Optional[str] = None,
    description: Optional[str] = None,
    max_workers: int = 4,
    timeout: Optional[float] = None,
    executor_type: ExecutorType = ExecutorType.THREAD,
    return_exceptions: bool = False,
    chunk_size: Optional[int] = None,
    ordered: bool = True,
    retry_on_failure: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
):
    """
    A decorator that enables concurrent execution of functions.

    Args:
        name (Optional[str]): Name for the concurrent operation
        description (Optional[str]): Description of the operation
        max_workers (int): Maximum number of worker threads/processes
        timeout (Optional[float]): Timeout in seconds for each task
        executor_type (ExecutorType): Type of executor (thread, process, async)
        return_exceptions (bool): Whether to return exceptions instead of raising
        chunk_size (Optional[int]): Size of chunks for batch processing
        ordered (bool): Whether to maintain order of results
        retry_on_failure (bool): Whether to retry failed tasks
        max_retries (int): Maximum number of retries per task
        retry_delay (float): Delay between retries in seconds

    Returns:
        Callable: Decorated function that can execute concurrently
    """

    if max_workers is None:
        max_workers = os.cpu_count()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        config = ConcurrentConfig(
            name=name or func.__name__,
            description=description
            or f"Concurrent execution of {func.__name__}",
            max_workers=max_workers,
            timeout=timeout,
            executor_type=executor_type,
            return_exceptions=return_exceptions,
            chunk_size=chunk_size,
            ordered=ordered,
            retry_on_failure=retry_on_failure,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        def _execute_single_task(
            task_args, task_kwargs, task_id=None
        ):
            """Execute a single task with retry logic."""
            start_time = time.time()

            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*task_args, **task_kwargs)
                    execution_time = time.time() - start_time
                    return ConcurrentResult(
                        value=result,
                        execution_time=execution_time,
                        worker_id=task_id,
                    )
                except Exception as e:
                    if (
                        attempt == config.max_retries
                        or not config.retry_on_failure
                    ):
                        execution_time = time.time() - start_time
                        if config.return_exceptions:
                            return ConcurrentResult(
                                exception=e,
                                execution_time=execution_time,
                                worker_id=task_id,
                            )
                        else:
                            raise
                    else:
                        logger.warning(
                            f"Task {task_id} failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}"
                        )
                        time.sleep(config.retry_delay * (2**attempt))

        def concurrent_execute(*args_list, **kwargs_list):
            """Execute the function concurrently with multiple argument sets."""
            if not args_list and not kwargs_list:
                raise ValueError(
                    "At least one set of arguments must be provided"
                )

            # Prepare tasks
            tasks = []
            if args_list:
                for args in args_list:
                    if isinstance(args, (list, tuple)):
                        tasks.append((args, {}))
                    else:
                        tasks.append(([args], {}))

            if kwargs_list:
                for kwargs in kwargs_list:
                    if isinstance(kwargs, dict):
                        tasks.append(((), kwargs))
                    else:
                        raise ValueError(
                            "kwargs_list must contain dictionaries"
                        )

            logger.info(
                f"Starting concurrent execution of {len(tasks)} tasks with {config.max_workers} workers"
            )
            start_time = time.time()

            try:
                if config.executor_type == ExecutorType.THREAD:
                    results = _execute_with_thread_pool(tasks)
                elif config.executor_type == ExecutorType.PROCESS:
                    results = _execute_with_process_pool(tasks)
                elif config.executor_type == ExecutorType.ASYNC:
                    results = _execute_with_async(tasks)
                else:
                    raise ValueError(
                        f"Unsupported executor type: {config.executor_type}"
                    )

                total_time = time.time() - start_time
                successful_tasks = sum(
                    1 for r in results if r.success
                )

                logger.info(
                    f"Completed {len(tasks)} tasks in {total_time:.3f}s "
                    f"({successful_tasks}/{len(tasks)} successful)"
                )

                return results

            except Exception as e:
                logger.error(f"Concurrent execution failed: {e}")
                raise

        def _execute_with_thread_pool(tasks):
            """Execute tasks using ThreadPoolExecutor."""
            results = []

            with ThreadPoolExecutor(
                max_workers=config.max_workers
            ) as executor:
                if config.ordered:
                    future_to_task = {
                        executor.submit(
                            _execute_single_task, task[0], task[1], i
                        ): i
                        for i, task in enumerate(tasks)
                    }

                    for future in as_completed(
                        future_to_task, timeout=config.timeout
                    ):
                        try:
                            result = future.result(
                                timeout=config.timeout
                            )
                            results.append(result)
                        except Exception as e:
                            if config.return_exceptions:
                                results.append(
                                    ConcurrentResult(exception=e)
                                )
                            else:
                                raise
                else:
                    futures = [
                        executor.submit(
                            _execute_single_task, task[0], task[1], i
                        )
                        for i, task in enumerate(tasks)
                    ]

                    for future in as_completed(
                        futures, timeout=config.timeout
                    ):
                        try:
                            result = future.result(
                                timeout=config.timeout
                            )
                            results.append(result)
                        except Exception as e:
                            if config.return_exceptions:
                                results.append(
                                    ConcurrentResult(exception=e)
                                )
                            else:
                                raise

            return results

        def _execute_with_process_pool(tasks):
            """Execute tasks using ProcessPoolExecutor."""
            results = []

            # Prepare task data for process execution
            task_data_list = []
            for i, task in enumerate(tasks):
                task_data = (
                    func,  # The function to execute
                    task[0],  # args
                    task[1],  # kwargs
                    i,  # task_id
                    config.max_retries,
                    config.retry_on_failure,
                    config.retry_delay,
                    config.return_exceptions,
                )
                task_data_list.append(task_data)

            with ProcessPoolExecutor(
                max_workers=config.max_workers
            ) as executor:
                if config.ordered:
                    future_to_task = {
                        executor.submit(
                            _execute_task_in_process, task_data
                        ): i
                        for i, task_data in enumerate(task_data_list)
                    }

                    for future in as_completed(
                        future_to_task, timeout=config.timeout
                    ):
                        try:
                            result = future.result(
                                timeout=config.timeout
                            )
                            results.append(result)
                        except Exception as e:
                            if config.return_exceptions:
                                results.append(
                                    ConcurrentResult(exception=e)
                                )
                            else:
                                raise
                else:
                    futures = [
                        executor.submit(
                            _execute_task_in_process, task_data
                        )
                        for task_data in task_data_list
                    ]

                    for future in as_completed(
                        futures, timeout=config.timeout
                    ):
                        try:
                            result = future.result(
                                timeout=config.timeout
                            )
                            results.append(result)
                        except Exception as e:
                            if config.return_exceptions:
                                results.append(
                                    ConcurrentResult(exception=e)
                                )
                            else:
                                raise

            return results

        async def _execute_with_async(tasks):
            """Execute tasks using asyncio."""

            async def _async_task(
                task_args, task_kwargs, task_id=None
            ):
                start_time = time.time()

                for attempt in range(config.max_retries + 1):
                    try:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None,
                            lambda: func(*task_args, **task_kwargs),
                        )
                        execution_time = time.time() - start_time
                        return ConcurrentResult(
                            value=result,
                            execution_time=execution_time,
                            worker_id=task_id,
                        )
                    except Exception as e:
                        if (
                            attempt == config.max_retries
                            or not config.retry_on_failure
                        ):
                            execution_time = time.time() - start_time
                            if config.return_exceptions:
                                return ConcurrentResult(
                                    exception=e,
                                    execution_time=execution_time,
                                    worker_id=task_id,
                                )
                            else:
                                raise
                        else:
                            logger.warning(
                                f"Async task {task_id} failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}"
                            )
                            await asyncio.sleep(
                                config.retry_delay * (2**attempt)
                            )

            semaphore = asyncio.Semaphore(config.max_workers)

            async def _limited_task(task_args, task_kwargs, task_id):
                async with semaphore:
                    return await _async_task(
                        task_args, task_kwargs, task_id
                    )

            tasks_coros = [
                _limited_task(task[0], task[1], i)
                for i, task in enumerate(tasks)
            ]

            if config.ordered:
                results = []
                for coro in asyncio.as_completed(tasks_coros):
                    try:
                        result = await coro
                        results.append(result)
                    except Exception as e:
                        if config.return_exceptions:
                            results.append(
                                ConcurrentResult(exception=e)
                            )
                        else:
                            raise
                return results
            else:
                return await asyncio.gather(
                    *tasks_coros,
                    return_exceptions=config.return_exceptions,
                )

        def concurrent_batch(
            items: List[Any],
            batch_size: Optional[int] = None,
            **kwargs,
        ) -> List[ConcurrentResult]:
            """Execute the function concurrently on a batch of items."""
            batch_size = batch_size or config.chunk_size or len(items)

            tasks = []
            for item in items:
                if isinstance(item, (list, tuple)):
                    tasks.append((item, kwargs))
                else:
                    tasks.append(([item], kwargs))

            return concurrent_execute(
                *[task[0] for task in tasks],
                **[task[1] for task in tasks],
            )

        def concurrent_map(
            items: List[Any], **kwargs
        ) -> List[ConcurrentResult]:
            """Map the function over a list of items concurrently."""
            return concurrent_batch(items, **kwargs)

        # Attach methods to the wrapper
        wrapper.concurrent_execute = concurrent_execute
        wrapper.concurrent_batch = concurrent_batch
        wrapper.concurrent_map = concurrent_map
        wrapper.config = config

        # Add metadata
        wrapper.__concurrent_config__ = config
        wrapper.__concurrent_enabled__ = True

        return wrapper

    return decorator


def concurrent_class_executor(
    name: Optional[str] = None,
    description: Optional[str] = None,
    max_workers: int = 4,
    timeout: Optional[float] = None,
    executor_type: ExecutorType = ExecutorType.THREAD,
    return_exceptions: bool = False,
    chunk_size: Optional[int] = None,
    ordered: bool = True,
    retry_on_failure: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    methods: Optional[List[str]] = None,
):
    """
    A decorator that enables concurrent execution for class methods.

    Args:
        name (Optional[str]): Name for the concurrent operation
        description (Optional[str]): Description of the operation
        max_workers (int): Maximum number of worker threads/processes
        timeout (Optional[float]): Timeout in seconds for each task
        executor_type (ExecutorType): Type of executor (thread, process, async)
        return_exceptions (bool): Whether to return exceptions instead of raising
        chunk_size (Optional[int]): Size of chunks for batch processing
        ordered (bool): Whether to maintain order of results
        retry_on_failure (bool): Whether to retry failed tasks
        max_retries (int): Maximum number of retries per task
        retry_delay (float): Delay between retries in seconds
        methods (Optional[List[str]]): List of method names to make concurrent

    Returns:
        Class: Class with concurrent execution capabilities
    """

    def decorator(cls):
        config = ConcurrentConfig(
            name=name or f"{cls.__name__}_concurrent",
            description=description
            or f"Concurrent execution for {cls.__name__}",
            max_workers=max_workers,
            timeout=timeout,
            executor_type=executor_type,
            return_exceptions=return_exceptions,
            chunk_size=chunk_size,
            ordered=ordered,
            retry_on_failure=retry_on_failure,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Get methods to make concurrent
        target_methods = methods or [
            name
            for name, method in inspect.getmembers(
                cls, inspect.isfunction
            )
            if not name.startswith("_")
        ]

        for method_name in target_methods:
            if hasattr(cls, method_name):
                original_method = getattr(cls, method_name)

                # Create concurrent version of the method
                concurrent_decorator = concurrent(
                    name=f"{cls.__name__}.{method_name}",
                    description=f"Concurrent execution of {cls.__name__}.{method_name}",
                    max_workers=config.max_workers,
                    timeout=config.timeout,
                    executor_type=config.executor_type,
                    return_exceptions=config.return_exceptions,
                    chunk_size=config.chunk_size,
                    ordered=config.ordered,
                    retry_on_failure=config.retry_on_failure,
                    max_retries=config.max_retries,
                    retry_delay=config.retry_delay,
                )

                # Apply the concurrent decorator to the method
                setattr(
                    cls,
                    method_name,
                    concurrent_decorator(original_method),
                )

        # Add class-level concurrent configuration
        cls.__concurrent_config__ = config
        cls.__concurrent_enabled__ = True

        return cls

    return decorator


# Convenience functions for common use cases
def thread_executor(**kwargs):
    """Convenience decorator for thread-based concurrent execution."""
    return concurrent(executor_type=ExecutorType.THREAD, **kwargs)


def process_executor(**kwargs):
    """Convenience decorator for process-based concurrent execution."""
    return concurrent(executor_type=ExecutorType.PROCESS, **kwargs)


def async_executor(**kwargs):
    """Convenience decorator for async-based concurrent execution."""
    return concurrent(executor_type=ExecutorType.ASYNC, **kwargs)


def batch_executor(batch_size: int = 10, **kwargs):
    """Convenience decorator for batch processing."""
    return concurrent(chunk_size=batch_size, **kwargs)
