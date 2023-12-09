from concurrent import futures
from typing import TypeVar

T = TypeVar("T")


def execute_futures_dict(fs_dict: dict[str, futures.Future[T]]) -> dict[str, T]:
    futures.wait(fs_dict.values(), timeout=None, return_when=futures.ALL_COMPLETED)

    return {key: future.result() for key, future in fs_dict.items()}
