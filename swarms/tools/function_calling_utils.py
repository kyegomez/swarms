import concurrent.futures
from typing import Any, Callable, Dict, List
from inspect import iscoroutinefunction
import asyncio


# Helper function to run an asynchronous function in a synchronous way
def run_async_function_in_sync(func: Callable, *args, **kwargs) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    coroutine = func(*args, **kwargs)
    return loop.run_until_complete(coroutine)


# Main omni function for parallel execution
def omni_parallel_function_caller(
    function_calls: List[Dict[str, Any]]
) -> List[Any]:
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_call = {}
        for call in function_calls:
            func = call["function"]
            args = call.get("args", ())
            kwargs = call.get("kwargs", {})

            if iscoroutinefunction(func):
                # Wrap and execute asynchronous function in a separate process
                future = executor.submit(
                    run_async_function_in_sync, func, *args, **kwargs
                )
            else:
                # Directly execute synchronous function in a thread
                future = executor.submit(func, *args, **kwargs)

            future_to_call[future] = call

        for future in concurrent.futures.as_completed(future_to_call):
            results.append(future.result())
    return results
