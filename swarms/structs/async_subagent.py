"""
Async subagent execution with background task registry,
recursive subagent trees, result aggregation, and fault tolerance.
"""

import uuid
import time
import threading
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    wait,
    ALL_COMPLETED,
    FIRST_COMPLETED,
)
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from loguru import logger


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubagentTask:
    """Tracks a single async subagent task."""

    id: str
    agent: Any
    task_str: str
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    future: Optional[Future] = None
    parent_id: Optional[str] = None
    depth: int = 0
    retries: int = 0
    max_retries: int = 0
    retry_on: Optional[List[Type[Exception]]] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class SubagentRegistry:
    """
    Manages async subagent tasks with status tracking,
    result aggregation, retry policies, and depth-limited recursion.
    """

    def __init__(
        self,
        max_depth: int = 3,
        max_workers: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self._tasks: Dict[str, SubagentTask] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def spawn(
        self,
        agent: Any,
        task: str,
        parent_id: Optional[str] = None,
        depth: int = 0,
        max_retries: int = 0,
        retry_on: Optional[List[Type[Exception]]] = None,
        fail_fast: bool = True,
    ) -> str:
        """
        Spawn an agent task in the background.

        Returns the task_id for tracking.
        Raises ValueError if depth exceeds max_depth.
        """
        if depth > self.max_depth:
            raise ValueError(
                f"Subagent depth {depth} exceeds max_depth {self.max_depth}"
            )

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        st = SubagentTask(
            id=task_id,
            agent=agent,
            task_str=task,
            parent_id=parent_id,
            depth=depth,
            max_retries=max_retries,
            retry_on=retry_on or [],
        )

        with self._lock:
            self._tasks[task_id] = st

        agent_name = getattr(agent, "agent_name", str(agent))
        logger.info(
            f"[SubagentRegistry] Spawned task {task_id} | agent={agent_name} | depth={depth}"
        )

        st.status = TaskStatus.RUNNING
        future = self._executor.submit(
            self._execute_task, st, fail_fast
        )
        st.future = future

        return task_id

    def _execute_task(self, st: SubagentTask, fail_fast: bool) -> Any:
        """Run the agent with retry logic."""
        agent_name = getattr(st.agent, "agent_name", str(st.agent))
        last_error = None

        for attempt in range(st.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(
                        f"[SubagentRegistry] Retry {attempt}/{st.max_retries} for task {st.id}"
                    )
                    st.retries = attempt

                result = st.agent.run(st.task_str)
                st.result = result
                st.status = TaskStatus.COMPLETED
                st.completed_at = time.time()
                logger.info(
                    f"[SubagentRegistry] Task {st.id} completed | agent={agent_name} | "
                    f"duration={st.completed_at - st.created_at:.2f}s"
                )
                return result

            except Exception as e:
                last_error = e
                should_retry = attempt < st.max_retries and (
                    not st.retry_on
                    or any(
                        isinstance(e, exc_type)
                        for exc_type in st.retry_on
                    )
                )
                if should_retry:
                    continue

                st.error = e
                st.status = TaskStatus.FAILED
                st.completed_at = time.time()
                logger.error(
                    f"[SubagentRegistry] Task {st.id} failed | agent={agent_name} | error={e}"
                )
                if fail_fast:
                    raise
                return None

        # Should not reach here, but handle edge case
        st.error = last_error
        st.status = TaskStatus.FAILED
        st.completed_at = time.time()
        if fail_fast:
            raise last_error
        return None

    def get_task(self, task_id: str) -> SubagentTask:
        """Get a task by ID."""
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")
        return self._tasks[task_id]

    def get_results(self) -> Dict[str, Any]:
        """Collect results from all completed tasks."""
        results = {}
        for task_id, st in self._tasks.items():
            if st.status == TaskStatus.COMPLETED:
                results[task_id] = st.result
            elif st.status == TaskStatus.FAILED:
                results[task_id] = st.error
        return results

    def cancel(self, task_id: str) -> bool:
        """Cancel a task if it hasn't completed yet."""
        st = self.get_task(task_id)
        if st.future and st.future.cancel():
            st.status = TaskStatus.CANCELLED
            st.completed_at = time.time()
            logger.info(
                f"[SubagentRegistry] Task {task_id} cancelled"
            )
            return True
        return False

    def gather(
        self,
        strategy: str = "wait_all",
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        Wait for tasks and return results.

        Args:
            strategy: "wait_all" or "wait_first"
            timeout: Max seconds to wait

        Returns:
            List of results (or exceptions for failed tasks)
        """
        # Collect already-completed results
        already_done = []
        pending_futures = {}
        for st in self._tasks.values():
            if st.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                already_done.append(st)
            elif st.future is not None:
                pending_futures[st.future] = st

        if not pending_futures:
            return [
                (
                    st.error
                    if st.status == TaskStatus.FAILED
                    else st.result
                )
                for st in already_done
            ]

        return_when = (
            FIRST_COMPLETED
            if strategy == "wait_first"
            else ALL_COMPLETED
        )
        done, _ = wait(
            pending_futures.keys(),
            timeout=timeout,
            return_when=return_when,
        )

        results = [
            st.error if st.status == TaskStatus.FAILED else st.result
            for st in already_done
        ]
        for future in done:
            try:
                result = future.result(timeout=0)
                results.append(result)
            except Exception as e:
                results.append(e)

        return results

    def shutdown(self):
        """Shut down the executor."""
        self._executor.shutdown(wait=False)
        logger.info("[SubagentRegistry] Shut down")

    @property
    def tasks(self) -> Dict[str, SubagentTask]:
        return dict(self._tasks)
