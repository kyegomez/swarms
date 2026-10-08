import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_cpu_cores() -> int:
    """
    Get the number of CPU cores available on the current system.

    Returns:
        int: Number of CPU cores, or None if undetectable.

    This function is cached for performance since the CPU count will not
    change during a program's execution.
    """
    return os.cpu_count()


@lru_cache(maxsize=1)
def max_workers_95_percent() -> int:
    """
    Get the number of worker threads to use for parallel execution.

    Returns:
        int: The number of worker threads to use, at least 1.
    """
    return max(1, int((os.cpu_count() or 1) * 0.95))
