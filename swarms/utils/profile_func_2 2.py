from functools import wraps
from loguru import logger
import tracemalloc
import psutil
import time
from typing import Callable, Any


def profile_all(func: Callable) -> Callable:
    """
    A decorator to profile memory usage, CPU usage, and I/O operations
    of a function and log the data using loguru.

    It combines tracemalloc for memory profiling, psutil for CPU and I/O operations,
    and measures execution time.

    Args:
        func (Callable): The function to be profiled.

    Returns:
        Callable: The wrapped function with profiling enabled.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Start memory tracking
        tracemalloc.start()

        # Get initial CPU stats
        process = psutil.Process()
        initial_cpu_times = process.cpu_times()

        # Get initial I/O stats if available
        try:
            initial_io_counters = process.io_counters()
            io_tracking_available = True
        except AttributeError:
            logger.warning(
                "I/O counters not available on this platform."
            )
            io_tracking_available = False

        # Start timing the function execution
        start_time = time.time()

        # Execute the function
        result = func(*args, **kwargs)

        # Stop timing
        end_time = time.time()
        execution_time = end_time - start_time

        # Get final CPU stats
        final_cpu_times = process.cpu_times()

        # Get final I/O stats if available
        if io_tracking_available:
            final_io_counters = process.io_counters()
            io_read_count = (
                final_io_counters.read_count
                - initial_io_counters.read_count
            )
            io_write_count = (
                final_io_counters.write_count
                - initial_io_counters.write_count
            )
        else:
            io_read_count = io_write_count = 0

        # Get memory usage statistics
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        # Calculate CPU usage
        cpu_usage = (
            final_cpu_times.user
            - initial_cpu_times.user
            + final_cpu_times.system
            - initial_cpu_times.system
        )

        # Log the data
        logger.info(f"Execution time: {execution_time:.4f} seconds")
        logger.info(f"CPU usage: {cpu_usage:.2f} seconds")
        if io_tracking_available:
            logger.info(
                f"I/O Operations - Read: {io_read_count}, Write: {io_write_count}"
            )
        logger.info("Top memory usage:")
        for stat in top_stats[:10]:
            logger.info(stat)

        # Stop memory tracking
        tracemalloc.stop()

        return result

    return wrapper
