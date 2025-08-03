import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_cpu_cores() -> int:
    return os.cpu_count()
