from __future__ import annotations

import logging
import os
import uuid
from typing import Iterable, List, Optional

from .task_queue import InMemoryTaskQueue, Task

logger = logging.getLogger(__name__)


class Planner:
    """Base planner class. Planners discover, plan, and create tasks, but do not execute them."""

    def __init__(self, task_queue=None, name: Optional[str] = None):
        self.name = name or f"planner-{uuid.uuid4().hex[:6]}"
        self.task_queue = task_queue or InMemoryTaskQueue()

    def explore_codebase(self) -> Iterable[str]:
        """Explore repository to identify areas of interest.

        Returns an iterable of strings representing files or modules.
        """
        root = os.getcwd()
        # Minimal default: list top-level files
        try:
            for entry in os.listdir(root):
                yield entry
        except Exception:
            return []

    def generate_tasks_from_goal(self, goal: str) -> List[Task]:
        """Generate high-level tasks from a project goal.

        Override in subclasses with model-driven logic.
        """
        # Lightweight fallback: return a single task
        t = Task(
            id=uuid.uuid4().hex,
            title=f"Work towards: {goal}",
            description=goal,
            priority=50,
        )
        return [t]

    def spawn_subplanner(self, area: str) -> "Planner":
        """Spawn a sub-planner for a focused area. Override if needed."""
        from .sub_planner import SubPlanner

        return SubPlanner(task_queue=self.task_queue, name=f"sub-{area}-{uuid.uuid4().hex[:4]}")
