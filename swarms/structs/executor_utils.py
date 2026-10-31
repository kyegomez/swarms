from typing import Optional

from swarms.telemetry.otel import ContextThreadPoolExecutor


class ReusableExecutorMixin:
    _executor: Optional[ContextThreadPoolExecutor] = None
    _executor_workers: int = 0

    def _get_executor(self, size: int) -> ContextThreadPoolExecutor:
        executor = self._executor
        if executor is None or size > self._executor_workers:
            executor = ContextThreadPoolExecutor(size)
            self._executor = executor
            self._executor_workers = size
        return executor

    def close(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
