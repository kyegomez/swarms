from __future__ import annotations

import logging
import uuid
from typing import List

from .planner import Planner
from .task_queue import Task

logger = logging.getLogger(__name__)


class SubPlanner(Planner):
    """Sub-planner that breaks down tasks into actionable items."""

    def __init__(self, task_queue=None, name: str = "sub-planner"):
        super().__init__(task_queue=task_queue, name=name)

    def breakdown_task(self, parent_task: Task) -> List[Task]:
        """Break a higher-level task into subtasks. Minimal heuristic fallback."""
        subtasks: List[Task] = []
        # Naive split: create two subtasks to encourage parallelization
        for i in range(2):
            tid = uuid.uuid4().hex
            subtasks.append(
                Task(
                    id=tid,
                    title=f"{parent_task.title} - subtask {i+1}",
                    description=f"Part {i+1} of: {parent_task.description}",
                    priority=parent_task.priority + i,
                    dependencies=[parent_task.id],
                )
            )
        return subtasks

    def plan_and_enqueue(self, parent_task: Task):
        subs = self.breakdown_task(parent_task)
        for s in subs:
            logger.info("Sub-planner %s enqueuing %s", self.name, s.title)
            self.task_queue.push_task(s)
