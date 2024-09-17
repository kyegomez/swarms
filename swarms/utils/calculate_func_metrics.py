import time
import psutil
from pydantic import BaseModel
from swarms.utils.loguru_logger import logger


class FunctionMetrics(BaseModel):
    execution_time: float
    memory_usage: float
    cpu_usage: float
    io_operations: int
    function_calls: int


def profile_func(func):
    """
    Decorator function that profiles the execution of a given function.

    Args:
        func: The function to be profiled.

    Returns:
        A wrapper function that profiles the execution of the given function and returns the result along with the metrics.

    """

    def wrapper(*args, **kwargs):
        # Record the initial time, memory usage, CPU usage, and I/O operations
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss
        start_cpu = psutil.cpu_percent()
        start_io = (
            psutil.disk_io_counters().read_count
            + psutil.disk_io_counters().write_count
        )

        # Call the function
        result = func(*args, **kwargs)

        # Record the final time, memory usage, CPU usage, and I/O operations
        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent()
        end_io = (
            psutil.disk_io_counters().read_count
            + psutil.disk_io_counters().write_count
        )

        # Calculate the execution time, memory usage, CPU usage, and I/O operations
        execution_time = end_time - start_time
        memory_usage = (end_mem - start_mem) / (
            1024**2
        )  # Convert bytes to MiB
        cpu_usage = end_cpu - start_cpu
        io_operations = end_io - start_io

        # Return the metrics as a FunctionMetrics object
        metrics = FunctionMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            io_operations=io_operations,
            function_calls=1,  # Each call to the function counts as one function call
        )

        json_data = metrics.model_dump_json(indent=4)

        logger.info(f"Function metrics: {json_data}")

        return result, metrics

    return wrapper
