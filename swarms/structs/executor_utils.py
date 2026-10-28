from typing import Optional

from swarms.telemetry.otel import ContextThreadPoolExecutor


class ReusableExecutorMixin:
    _executor: Optional[ContextThreadPoolExecutor] = None
    _executor_workers: int = 0

    def _get_executor(
        self, max_workers: int
    ) -> ContextThreadPoolExecutor:
        executor = self._executor
        if executor is None or max_workers > self._executor_workers:
            executor = ContextThreadPoolExecutor(max_workers)
            self._executor = executor
            self._executor_workers = max_workers
        return executor

    def close(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
