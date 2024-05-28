import timeit
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def optimized_loop(
    data: Iterable[T],
    operation: Callable[[T], R],
    condition: Optional[Callable[[T], bool]] = None,
) -> List[R]:
    """
    Perform an optimized loop over the input data, applying an operation to each element.
    Optionally, filter elements based on a condition before applying the operation.

    Args:
        data (Iterable[T]): The input data to be processed. Can be any iterable type.
        operation (Callable[[T], R]): The operation to be applied to each element.
        condition (Optional[Callable[[T], bool]]): An optional condition to filter elements before applying the operation.

    Returns:
        List[R]: The result of applying the operation to the filtered elements.
    """
    if condition is not None:
        return [operation(x) for x in data if condition(x)]
    else:
        return [operation(x) for x in data]


# Sample data, operation, and condition for benchmarking
data = list(range(1000000))


def operation(x):
    return x * x


def condition(x):
    return x % 2 == 0


# Define a traditional loop for comparison
def traditional_loop(data: Iterable[int]) -> List[int]:
    result = []
    for x in data:
        if x % 2 == 0:
            result.append(x * x)
    return result


# Define a benchmarking function
def benchmark():
    # Time the execution of the optimized loop
    optimized_time = timeit.timeit(
        stmt="optimized_loop(data, operation, condition)",
        setup="from __main__ import optimized_loop, data, operation, condition",
        globals=globals(),
        number=10,
    )

    print(f"Optimized loop execution time: {optimized_time:.4f} seconds")

    # Time the execution of the traditional loop for comparison
    traditional_time = timeit.timeit(
        stmt="traditional_loop(data)",
        setup="from __main__ import traditional_loop, data",
        globals=globals(),
        number=10,
    )

    print(
        f"Traditional loop execution time: {traditional_time:.4f} seconds"
    )


# Run the benchmark
if __name__ == "__main__":
    benchmark()
