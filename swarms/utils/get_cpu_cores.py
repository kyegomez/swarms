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
