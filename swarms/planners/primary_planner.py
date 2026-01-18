from __future__ import annotations

import logging
import uuid
from typing import Iterable, List, Optional

from .planner import Planner
from .task_queue import Task

logger = logging.getLogger(__name__)


class PrimaryPlanner(Planner):
    """Primary planner that continuously explores the codebase and creates high-level tasks."""

    def __init__(self, task_queue=None, name: Optional[str] = None):
        super().__init__(task_queue=task_queue, name=name or "primary-planner")

    def run_once(self, project_goals: Iterable[str]):
        """Perform a single planning cycle: explore and enqueue tasks."""
        for goal in project_goals:
            tasks: List[Task] = self.generate_tasks_from_goal(goal)
            for t in tasks:
                logger.info("Enqueueing task %s", t.title)
                self.task_queue.push_task(t)

    def monitor_progress(self):
        # Placeholder for observability hooks
        pending = self.task_queue.list_pending()
        logger.info("Pending tasks: %d", len(pending))
