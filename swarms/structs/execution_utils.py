import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from swarms.telemetry.otel import ContextThreadPoolExecutor


def batched_run(
    func: Callable,
    tasks: Sequence[Any],
    *args: Any,
    img: Optional[Union[str, Sequence[Optional[str]]]] = None,
    imgs: Optional[Sequence[Optional[str]]] = None,
    max_workers: Optional[int] = None,
    return_agent_output_dict: bool = False,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> Union[List[Any], Dict[Any, Any]]:
    """
    Run ``func`` once per task and collect the results.

    Args:
        func (Callable): The callable to invoke per task, usually a bound
            ``run`` method. Called as ``func(task, *args, **kwargs)``.
        tasks (Sequence[Any]): The tasks, each passed as the first argument.
        *args: Extra positional arguments forwarded to ``func``.
        img (Optional[str | Sequence[Optional[str]]]): One image
            broadcast to every task as ``img=``. A sequence is accepted and
            treated as ``imgs`` - callers spell this parameter both ways.
            Only forwarded when set, so callables without the parameter
            still work. Mutually exclusive with ``imgs``.
        imgs (Optional[Sequence[Optional[str]]]): One image per task, paired
            by position. Must be the same length as ``tasks``.
        max_workers (Optional[int]): Run concurrently with this many threads.
            ``None`` (the default) runs sequentially. Results stay in task
            order either way.
        return_agent_output_dict (bool): Return ``{task: result}`` instead of
            a list. Tasks must be hashable, and duplicates collapse.
        return_exceptions (bool): Return the raised exception in place of a
            result rather than propagating it, as ``asyncio.gather`` does.
        **kwargs: Extra keyword arguments forwarded to ``func``.

    Returns:
        Union[List[Any], Dict[Any, Any]]: Results in task order, or keyed by
        task when ``return_agent_output_dict`` is set.

    Raises:
        TypeError: If ``func`` is not callable.
        ValueError: If ``max_workers`` is less than 1, if both ``img`` and
            ``imgs`` are given, or if ``imgs`` is not one per task.
        Exception: Anything ``func`` raises, unless ``return_exceptions``.
    """
    if not callable(func):
        raise TypeError(
            f"func must be callable, got {type(func).__name__}"
        )
    if max_workers is not None and max_workers < 1:
        raise ValueError(
            f"max_workers must be >= 1, got {max_workers}"
        )

    if img is not None and imgs is not None:
        raise ValueError(
            "Pass either img (one for every task) or imgs (one per "
            "task), not both."
        )

    # A sequence in img means per-task images: callers pass a list here
    # (some even type the parameter that way), and broadcasting it would
    # hand every task the whole list as its image.
    if imgs is None and isinstance(img, (list, tuple)):
        imgs, img = list(img), None

    tasks = list(tasks)
    if not tasks:
        return {} if return_agent_output_dict else []

    if imgs is not None:
        imgs = list(imgs)
        # Length-checked rather than zipped: zip would silently drop the
        # tail and run fewer tasks than the caller asked for.
        if len(imgs) != len(tasks):
            raise ValueError(
                f"Got {len(tasks)} tasks and {len(imgs)} images; pass "
                "one image per task, or omit imgs entirely."
            )
    elif img is not None:
        imgs = [img] * len(tasks)

    def call(index: int, task: Any) -> Any:
        per_task = dict(kwargs)
        if imgs is not None and imgs[index] is not None:
            per_task["img"] = imgs[index]
        if return_exceptions:
            try:
                return func(task, *args, **per_task)
            except Exception as exc:
                return exc
        return func(task, *args, **per_task)

    if max_workers is None:
        results = [call(i, task) for i, task in enumerate(tasks)]
    else:
        # ContextThreadPoolExecutor, not the plain one: it carries the
        # tracing context in, so worker spans stay nested under the run.
        with ContextThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Futures are read in submission order, so element i is always
            # the result for tasks[i] even when they finish out of order.
            futures = [
                executor.submit(call, i, task)
                for i, task in enumerate(tasks)
            ]
            results = [future.result() for future in futures]

    if return_agent_output_dict:
        return dict(zip(tasks, results))
    return results


def run_concurrently(
    func: Callable,
    tasks: Sequence[Any],
    *args: Any,
    max_workers: Optional[int] = None,
    **kwargs: Any,
) -> Union[List[Any], Dict[Any, Any]]:
    """
    Run ``func`` once per task across a thread pool.

    A thin front for :func:`batched_run` for callers who want concurrency by
    default rather than opting in with ``max_workers``. Results come back in
    task order, and the tracing context follows each task into its worker.

    Args:
        func (Callable): The callable to invoke per task.
        tasks (Sequence[Any]): The tasks, each passed as the first argument.
        *args: Extra positional arguments forwarded to ``func``.
        max_workers (Optional[int]): Thread count. Defaults to
            ``os.cpu_count()``, which is what the hand-written versions used.
        **kwargs: Extra keyword arguments forwarded to :func:`batched_run`,
            including ``img``, ``return_agent_output_dict`` and
            ``return_exceptions``.

    Returns:
        Union[List[Any], Dict[Any, Any]]: Results in task order.
    """
    return batched_run(
        func,
        tasks,
        *args,
        max_workers=max_workers or os.cpu_count() or 1,
        **kwargs,
    )
