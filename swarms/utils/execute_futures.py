from concurrent import futures
from concurrent.futures import Future
from typing import Dict, TypeVar

T = TypeVar("T")


def execute_futures_dict(
    fs_dict: Dict[str, Future[T]],
) -> Dict[str, T]:
    """Execute a dictionary of futures and return the results.

    Args:
        fs_dict (dict[str, futures.Future[T]]): _description_

    Returns:
        dict[str, T]: _description_

    Example:
    >>> import concurrent.futures
    >>> import time
    >>> import random
    >>> import swarms.utils.futures
    >>> def f(x):
    ...     time.sleep(random.random())
    ...     return x
    >>> with concurrent.futures.ThreadPoolExecutor() as executor:
    ...     fs_dict = {
    ...         str(i): executor.submit(f, i)
    ...         for i in range(10)
    ...     }
    ...     print(swarms.utils.futures.execute_futures_dict(fs_dict))
    {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

    """
    futures.wait(
        fs_dict.values(),
        timeout=None,
        return_when=futures.ALL_COMPLETED,
    )

    return {key: future.result() for key, future in fs_dict.items()}
