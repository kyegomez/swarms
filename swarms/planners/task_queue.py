from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    priority: int = 50
    dependencies: Sequence[str] = field(default_factory=list)
    status: str = "pending"  # pending, in-progress, completed, failed
    owner: Optional[str] = None
    complexity: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "Task":
        data = json.loads(s)
        return Task(**data)


class InMemoryTaskQueue:
    """Simple in-memory task queue for development and tests.

    Designed to emulate atomic pop via threading.Lock.
    """

    def __init__(self) -> None:
        self._pending: List[str] = []
        self._processing: List[str] = []
        self._tasks: Dict[str, str] = {}
        self._lock = threading.Lock()

    def push_task(self, task: Task) -> None:
        with self._lock:
            self._tasks[task.id] = task.to_json()
            self._pending.append(task.id)

    def pop_task_atomic(self) -> Optional[Task]:
        with self._lock:
            if not self._pending:
                return None
            tid = self._pending.pop(0)
            self._processing.append(tid)
            raw = self._tasks.get(tid)
            if raw is None:
                return None
            task = Task.from_json(raw)
            task.status = "in-progress"
            task.updated_at = time.time()
            self._tasks[tid] = task.to_json()
            return task

    def ack_task(self, task_id: str, success: bool = True) -> None:
        with self._lock:
            if task_id in self._processing:
                self._processing.remove(task_id)
            raw = self._tasks.get(task_id)
            if not raw:
                return
            task = Task.from_json(raw)
            task.status = "completed" if success else "failed"
            task.updated_at = time.time()
            self._tasks[task_id] = task.to_json()

    def get_task(self, task_id: str) -> Optional[Task]:
        raw = self._tasks.get(task_id)
        if not raw:
            return None
        return Task.from_json(raw)

    def list_pending(self) -> List[Task]:
        with self._lock:
            return [Task.from_json(self._tasks[tid]) for tid in list(self._pending)]


class RedisTaskQueue:
    """Minimal Redis-backed queue. Requires `redis` package and a running Redis.

    We keep implementation small: pending list, processing list, and per-task hash.
    """

    def __init__(self, redis_client):
        self.r = redis_client
        self.pending_key = "planner:pending"
        self.processing_key = "planner:processing"
        self.task_key_prefix = "planner:task:"

    def _task_key(self, task_id: str) -> str:
        return f"{self.task_key_prefix}{task_id}"

    def push_task(self, task: Task) -> None:
        self.r.set(self._task_key(task.id), task.to_json())
        self.r.rpush(self.pending_key, task.id)

    def pop_task_atomic(self, timeout: int = 0) -> Optional[Task]:
        # Use RPOPLPUSH to move from pending -> processing atomically
        tid = None
        if timeout and hasattr(self.r, "brpoplpush"):
            tid = self.r.brpoplpush(self.pending_key, self.processing_key, timeout)
        else:
            tid = self.r.rpoplpush(self.pending_key, self.processing_key)
        if not tid:
            return None
        raw = self.r.get(self._task_key(tid))
        if not raw:
            return None
        task = Task.from_json(raw)
        task.status = "in-progress"
        task.updated_at = time.time()
        self.r.set(self._task_key(task.id), task.to_json())
        return task

    def ack_task(self, task_id: str, success: bool = True) -> None:
        # Remove from processing list and update status
        self.r.lrem(self.processing_key, 0, task_id)
        raw = self.r.get(self._task_key(task_id))
        if not raw:
            return
        task = Task.from_json(raw)
        task.status = "completed" if success else "failed"
        task.updated_at = time.time()
        self.r.set(self._task_key(task.id), task.to_json())

    def get_task(self, task_id: str) -> Optional[Task]:
        raw = self.r.get(self._task_key(task_id))
        if not raw:
            return None
        return Task.from_json(raw)
