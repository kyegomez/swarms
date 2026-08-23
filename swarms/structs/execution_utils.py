import concurrent.futures
from typing import List, Callable, Dict, Any


def batched_run(
    func: Callable,
    tasks: List[str],
    img: str = None,
    return_agent_output_dict: bool = False,
    max_workers: int = None,
) -> Dict[str, Any]:
    """
    Runs a list of tasks concurrently and returns their outputs as a dictionary mapping each task to its result.
    Optionally returns a more detailed agent output dict if specified.
    """
    from collections.abc import Callable as _Callable  # type: ignore

    if not isinstance(func, _Callable):
        raise ValueError("func must be callable")
    results_dict = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_task = {
            executor.submit(func, task, img): task for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if return_agent_output_dict and isinstance(
                    result, dict
                ):
                    results_dict[task] = result
                else:
                    # If it's not a dict, just store the output as string
                    results_dict[task] = str(result)
            except Exception as exc:
                results_dict[task] = f"Generated an exception: {exc}"
    return results_dict
