import concurrent.futures
import json
import os
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from swarms.prompts.planner_worker_prompts import (
    JUDGE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    WORKER_SYSTEM_PROMPT,
)
from swarms.schemas.planner_worker_schemas import (
    CycleVerdict,
    PlannerTask,
    PlannerTaskSpec,
    PlannerTaskStatus,
    TaskPriority,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.tools.base_tool import BaseTool
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.output_types import OutputType


class TaskQueue:
    """Thread-safe in-memory task queue with optimistic concurrency.

    Uses a dict of PlannerTask objects protected by a threading.Lock.
    Workers claim tasks atomically via claim(). The version field on
    each task enables optimistic concurrency for start/complete/fail.
    """

    def __init__(self):
        self._tasks: Dict[str, PlannerTask] = {}
        self._lock = threading.Lock()

    def add_task(self, task: PlannerTask) -> str:
        """Add a task to the queue. Returns the task ID."""
        with self._lock:
            self._tasks[task.id] = task
            logger.info(
                f"[TaskQueue] Added task {task.id}: {task.title}"
            )
            return task.id

    def add_tasks(self, tasks: List[PlannerTask]) -> List[str]:
        """Bulk add tasks. Returns list of task IDs."""
        ids = []
        with self._lock:
            for task in tasks:
                self._tasks[task.id] = task
                ids.append(task.id)
        logger.info(f"[TaskQueue] Added {len(ids)} tasks")
        return ids

    def claim(self, worker_name: str) -> Optional[PlannerTask]:
        """Atomically claim the highest-priority available task.

        A task is available if:
        1. status == PENDING
        2. All depends_on task IDs are in COMPLETED status

        Returns None if no tasks are available.
        """
        with self._lock:
            completed_ids = {
                tid
                for tid, t in self._tasks.items()
                if t.status == PlannerTaskStatus.COMPLETED
            }

            available = []
            for task in self._tasks.values():
                if task.status != PlannerTaskStatus.PENDING:
                    continue
                if all(
                    dep_id in completed_ids
                    for dep_id in task.depends_on
                ):
                    available.append(task)

            if not available:
                return None

            # Highest priority first, then oldest first
            available.sort(
                key=lambda t: (-t.priority.value, t.created_at)
            )

            chosen = available[0]
            chosen.status = PlannerTaskStatus.CLAIMED
            chosen.assigned_worker = worker_name
            chosen.version += 1

            return chosen.model_copy()

    def start(self, task_id: str, expected_version: int) -> bool:
        """Mark a claimed task as RUNNING. Returns False on version mismatch."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.version != expected_version:
                return False
            if task.status != PlannerTaskStatus.CLAIMED:
                return False
            task.status = PlannerTaskStatus.RUNNING
            task.version += 1
            return True

    def complete(
        self, task_id: str, result: str, expected_version: int
    ) -> bool:
        """Mark a running task as COMPLETED with result."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.version != expected_version:
                return False
            if task.status != PlannerTaskStatus.RUNNING:
                return False
            task.status = PlannerTaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            task.version += 1
            return True

    def fail(
        self, task_id: str, error: str, expected_version: int
    ) -> bool:
        """Mark a running task as FAILED. Resets to PENDING if retries remain."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.version != expected_version:
                return False
            if task.status != PlannerTaskStatus.RUNNING:
                return False

            task.retries += 1
            task.error = error

            if task.retries <= task.max_retries:
                # Reset for retry
                task.status = PlannerTaskStatus.PENDING
                task.assigned_worker = None
                logger.info(
                    f"[TaskQueue] Task {task_id} failed, retrying ({task.retries}/{task.max_retries})"
                )
            else:
                task.status = PlannerTaskStatus.FAILED
                task.completed_at = time.time()
                logger.warning(
                    f"[TaskQueue] Task {task_id} permanently failed: {error}"
                )

            task.version += 1
            return True

    def cancel(self, task_id: str) -> bool:
        """Cancel a task if not yet completed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in (
                PlannerTaskStatus.COMPLETED,
                PlannerTaskStatus.CANCELLED,
            ):
                return False
            task.status = PlannerTaskStatus.CANCELLED
            task.version += 1
            return True

    def clear(self) -> int:
        """Remove all tasks from the queue. Returns the count of tasks removed.

        Used for fresh starts when the judge determines that accumulated
        drift requires a complete restart rather than incremental gap-filling.
        """
        with self._lock:
            count = len(self._tasks)
            self._tasks.clear()
            if count:
                logger.info(
                    f"[TaskQueue] Cleared {count} tasks (fresh start)"
                )
            return count

    def clear_non_terminal(self) -> int:
        """Remove only pending/claimed/running tasks, preserving completed results.

        Used between cycles when the judge determines that completed work
        is still valuable but remaining work should be replanned.
        """
        with self._lock:
            terminal = {
                PlannerTaskStatus.COMPLETED,
                PlannerTaskStatus.FAILED,
                PlannerTaskStatus.CANCELLED,
            }
            to_remove = [
                tid
                for tid, t in self._tasks.items()
                if t.status not in terminal
            ]
            for tid in to_remove:
                del self._tasks[tid]
            if to_remove:
                logger.info(
                    f"[TaskQueue] Cleared {len(to_remove)} non-terminal tasks"
                )
            return len(to_remove)

    def get_task(self, task_id: str) -> Optional[PlannerTask]:
        """Get a read-only copy of a task by ID."""
        with self._lock:
            task = self._tasks.get(task_id)
            return task.model_copy() if task else None

    def get_all_tasks(self) -> List[PlannerTask]:
        """Get read-only copies of all tasks."""
        with self._lock:
            return [t.model_copy() for t in self._tasks.values()]

    def get_pending_count(self) -> int:
        """Count of tasks in PENDING status with satisfied dependencies."""
        with self._lock:
            completed_ids = {
                tid
                for tid, t in self._tasks.items()
                if t.status == PlannerTaskStatus.COMPLETED
            }
            return sum(
                1
                for t in self._tasks.values()
                if t.status == PlannerTaskStatus.PENDING
                and all(
                    dep in completed_ids for dep in t.depends_on
                )
            )

    def get_completed_count(self) -> int:
        with self._lock:
            return sum(
                1
                for t in self._tasks.values()
                if t.status == PlannerTaskStatus.COMPLETED
            )

    def get_failed_count(self) -> int:
        with self._lock:
            return sum(
                1
                for t in self._tasks.values()
                if t.status == PlannerTaskStatus.FAILED
            )

    def is_all_done(self) -> bool:
        """True if all tasks are in a terminal state."""
        terminal = {
            PlannerTaskStatus.COMPLETED,
            PlannerTaskStatus.FAILED,
            PlannerTaskStatus.CANCELLED,
        }
        with self._lock:
            if not self._tasks:
                return True
            return all(
                t.status in terminal for t in self._tasks.values()
            )

    def get_results_summary(self) -> Dict[str, str]:
        """Return {task_id: result} for all completed tasks."""
        with self._lock:
            return {
                tid: t.result
                for tid, t in self._tasks.items()
                if t.status == PlannerTaskStatus.COMPLETED
                and t.result is not None
            }

    def get_dependency_results(self, task_id: str) -> str:
        """Get formatted results from all completed tasks that this task depends on."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or not task.depends_on:
                return ""
            parts = []
            for dep_id in task.depends_on:
                dep = self._tasks.get(dep_id)
                if (
                    dep
                    and dep.status == PlannerTaskStatus.COMPLETED
                    and dep.result
                ):
                    parts.append(
                        f"[Result from: {dep.title}]\n{dep.result}"
                    )
            return "\n\n".join(parts)

    def get_status(self) -> Dict[str, Any]:
        """Return a structured status report of the task queue."""
        with self._lock:
            status_counts: Dict[str, int] = {}
            task_details = []
            for t in self._tasks.values():
                status_counts[t.status.value] = (
                    status_counts.get(t.status.value, 0) + 1
                )
                task_details.append(
                    {
                        "id": t.id,
                        "title": t.title,
                        "status": t.status.value,
                        "assigned_worker": t.assigned_worker,
                        "priority": t.priority.name,
                        "retries": t.retries,
                    }
                )
            total = len(self._tasks)
            completed = status_counts.get("completed", 0)
            return {
                "total": total,
                "progress": f"{completed}/{total}",
                "status_counts": status_counts,
                "tasks": task_details,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._tasks)


class WorkerPool:
    """Manages a pool of worker agents that claim and execute tasks from a TaskQueue.

    Each worker runs in a ThreadPoolExecutor thread, independently claiming
    and executing tasks until the queue is drained or timeout is reached.
    Workers do not coordinate with each other — they only interact with
    the shared TaskQueue through atomic claim/start/complete/fail operations.
    """

    def __init__(
        self,
        agents: List[Agent],
        task_queue: TaskQueue,
        conversation: Conversation,
        max_workers: Optional[int] = None,
        poll_interval: float = 0.1,
        task_timeout: Optional[float] = None,
    ):
        self.agents = agents
        self.task_queue = task_queue
        self.conversation = conversation
        self.max_workers = max_workers or min(
            len(agents), os.cpu_count() or 4
        )
        self.poll_interval = poll_interval
        self.task_timeout = task_timeout
        self._stop_event = threading.Event()

    def run(
        self, timeout: Optional[float] = None
    ) -> Dict[str, str]:
        """Run all workers until queue is drained or timeout.

        Returns dict of {task_id: result} for completed tasks.
        """
        self._stop_event.clear()
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = []
            for agent in self.agents:
                future = executor.submit(
                    self._worker_loop, agent, timeout, start_time
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[WorkerPool] Worker failed: {e}")

        return self.task_queue.get_results_summary()

    def stop(self):
        """Signal all workers to stop."""
        self._stop_event.set()

    def _worker_loop(
        self,
        agent: Agent,
        timeout: Optional[float],
        start_time: float,
    ):
        """Main loop for a single worker agent."""
        worker_name = getattr(
            agent, "agent_name", str(id(agent))
        )

        while not self._stop_event.is_set():
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.info(
                    f"[WorkerPool] Worker {worker_name} timed out"
                )
                break

            # Check if all done
            if self.task_queue.is_all_done():
                break

            # Try to claim a task
            task = self.task_queue.claim(worker_name)
            if task is None:
                time.sleep(self.poll_interval)
                continue

            # Transition to RUNNING
            if not self.task_queue.start(task.id, task.version):
                continue

            logger.info(
                f"[WorkerPool] Worker {worker_name} executing: {task.title}"
            )

            try:
                # Reset agent memory so prior tasks don't leak into this one
                agent.short_memory = agent.short_memory_init()

                # Build context: worker focus prompt + task description + dependency results
                dep_context = self.task_queue.get_dependency_results(
                    task.id
                )
                context_parts = [
                    WORKER_SYSTEM_PROMPT.strip(),
                    f"\nTask: {task.title}",
                    f"Description: {task.description}",
                ]
                if dep_context:
                    context_parts.append(
                        f"\nContext from prerequisite tasks:\n{dep_context}"
                    )
                context = "\n".join(context_parts)

                # Run with per-task timeout to detect stuck workers
                if self.task_timeout:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=1
                    ) as task_executor:
                        future = task_executor.submit(
                            agent.run, task=context
                        )
                        try:
                            result = future.result(
                                timeout=self.task_timeout
                            )
                        except concurrent.futures.TimeoutError:
                            raise TimeoutError(
                                f"Task execution exceeded {self.task_timeout}s timeout"
                            )
                else:
                    result = agent.run(task=context)

                current = self.task_queue.get_task(task.id)
                if current and self.task_queue.complete(
                    task.id, result, current.version
                ):
                    self.conversation.add(
                        role=worker_name,
                        content=f"[Task: {task.title}]\n{result}",
                    )
                    logger.info(
                        f"[WorkerPool] Worker {worker_name} completed: {task.title}"
                    )
                else:
                    logger.warning(
                        f"[WorkerPool] Version conflict completing {task.id}"
                    )

            except Exception as e:
                logger.error(
                    f"[WorkerPool] Worker {worker_name} failed on {task.id}: {e}"
                )
                current = self.task_queue.get_task(task.id)
                if current:
                    self.task_queue.fail(
                        task.id, str(e), current.version
                    )


class PlannerWorkerSwarm:
    """A swarm that separates planning from execution.

    Implements the planner-worker architecture described in Cursor's
    "Scaling long-running autonomous coding" research. Key principles:

    - **Separation of concerns**: Planners only plan, workers only execute.
    - **No worker coordination**: Workers interact only with the task queue.
    - **Optimistic concurrency**: No locks — workers claim tasks atomically.
    - **Judge-driven cycles**: A judge evaluates results and decides whether
      to continue, fill gaps, or trigger a fresh start to combat drift.
    - **Prompts matter**: Worker and planner behavior is controlled through
      focused system prompts that reinforce role boundaries.

    Architecture per cycle:
        1. Planner agent(s) analyze the task and produce sub-tasks into a TaskQueue
        2. Worker agents claim tasks from the queue and execute them concurrently
        3. A judge agent evaluates the cycle results
        4. If not complete, the planner gets feedback and produces new tasks
        5. If drift is detected, the judge triggers a fresh start

    Args:
        name: Name of the swarm.
        description: Description of purpose.
        agents: Worker agents that execute tasks.
        max_loops: Maximum planner-worker-judge cycles.
        planner_model_name: Model for the planner agent.
        judge_model_name: Model for the judge agent.
        max_planner_depth: Max recursive sub-planner depth (1 = no sub-planners).
        worker_timeout: Max seconds for worker pool per cycle.
        task_timeout: Max seconds per individual task execution.
        max_workers: Max concurrent worker threads.
        output_type: Output format for the final result.
        autosave: Whether to save conversation history.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        name: str = "PlannerWorkerSwarm",
        description: str = "A planner-worker execution swarm",
        agents: Optional[List[Union[Agent, Callable]]] = None,
        max_loops: int = 1,
        planner_model_name: str = "gpt-4o-mini",
        judge_model_name: str = "gpt-4o-mini",
        max_planner_depth: int = 1,
        worker_timeout: Optional[float] = None,
        task_timeout: Optional[float] = None,
        max_workers: Optional[int] = None,
        output_type: OutputType = "dict-all-except-first",
        autosave: bool = False,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents or []
        self.max_loops = max_loops
        self.planner_model_name = planner_model_name
        self.judge_model_name = judge_model_name
        self.max_planner_depth = max_planner_depth
        self.worker_timeout = worker_timeout
        self.task_timeout = task_timeout
        self.max_workers = max_workers
        self.output_type = output_type
        self.autosave = autosave
        self.verbose = verbose

        # Internal state
        self.task_queue = TaskQueue()
        self.conversation = Conversation(time_enabled=False)
        self._original_task: Optional[str] = None

        self._reliability_checks()

    def _reliability_checks(self):
        if not self.agents or len(self.agents) == 0:
            raise ValueError(
                "PlannerWorkerSwarm requires at least one worker agent."
            )
        if self.max_loops <= 0:
            raise ValueError("max_loops must be > 0")

    def _create_planner_agent(
        self,
        name: str = "Planner",
    ) -> Agent:
        """Create a planner agent configured for structured task output."""
        schema = BaseTool().base_model_to_dict(PlannerTaskSpec)

        return Agent(
            agent_name=name,
            agent_description="Creates plans and decomposes tasks into concrete work items",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            model_name=self.planner_model_name,
            max_loops=1,
            base_model=PlannerTaskSpec,
            tools_list_dictionary=[schema],
            output_type="dict-all-except-first",
        )

    def _parse_structured_output(self, output: Any, model_class):
        """Parse structured output from an agent, handling various output formats.

        Follows the same pattern as HierarchicalSwarm.parse_orders().
        """
        try:
            if isinstance(output, model_class):
                return output

            if isinstance(output, dict):
                return model_class(**output)

            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict):
                        # Conversation format with role/content
                        if "content" in item and isinstance(
                            item["content"], list
                        ):
                            for content_item in item["content"]:
                                if (
                                    isinstance(content_item, dict)
                                    and "function" in content_item
                                ):
                                    function_data = content_item[
                                        "function"
                                    ]
                                    if (
                                        "arguments"
                                        in function_data
                                    ):
                                        try:
                                            args = json.loads(
                                                function_data[
                                                    "arguments"
                                                ]
                                            )
                                            return model_class(
                                                **args
                                            )
                                        except (
                                            json.JSONDecodeError,
                                            TypeError,
                                        ):
                                            pass
                        # Direct function call format
                        elif "function" in item:
                            function_data = item["function"]
                            if "arguments" in function_data:
                                try:
                                    args = json.loads(
                                        function_data["arguments"]
                                    )
                                    return model_class(**args)
                                except (
                                    json.JSONDecodeError,
                                    TypeError,
                                ):
                                    pass
                        # Try direct dict parse
                        try:
                            return model_class(**item)
                        except (TypeError, ValueError):
                            pass

            # Try parsing as JSON string
            if isinstance(output, str):
                try:
                    data = json.loads(output)
                    return model_class(**data)
                except (json.JSONDecodeError, TypeError):
                    pass

            raise ValueError(
                f"Unable to parse output as {model_class.__name__}: {type(output)}"
            )

        except Exception as e:
            logger.error(
                f"[PlannerWorkerSwarm] Failed to parse output: {e}\n"
                f"[TRACE] {traceback.format_exc()}"
            )
            raise

    def _run_planner(
        self,
        task: str,
        depth: int = 0,
        parent_task_id: Optional[str] = None,
    ) -> List[PlannerTask]:
        """Run a planner and add produced tasks to the queue.

        Args:
            task: The task description to plan for.
            depth: Current recursion depth for sub-planners.
            parent_task_id: ID of parent task if sub-planning.

        Returns:
            List of PlannerTask objects added to the queue.
        """
        planner_name = (
            "Planner" if depth == 0 else f"SubPlanner-{depth}"
        )
        planner = self._create_planner_agent(name=planner_name)

        logger.info(
            f"[PlannerWorkerSwarm] Running {planner_name} (depth={depth})"
        )

        raw_output = planner.run(task=task)

        spec = self._parse_structured_output(
            raw_output, PlannerTaskSpec
        )

        self.conversation.add(
            role=planner_name, content=spec.plan
        )

        # Convert PlannerTaskOutput items to PlannerTask objects
        added_tasks = []
        title_to_id: Dict[str, str] = {}

        # First pass: create tasks and build title->id map
        task_pairs: List[tuple[PlannerTask, List[str]]] = []
        for task_output in spec.tasks:
            priority = TaskPriority(
                max(0, min(3, task_output.priority))
            )
            ptask = PlannerTask(
                title=task_output.title,
                description=task_output.description,
                priority=priority,
                parent_task_id=parent_task_id,
            )
            title_to_id[task_output.title] = ptask.id
            task_pairs.append(
                (ptask, task_output.depends_on_titles)
            )

        # Second pass: resolve dependency titles to IDs and add to queue
        for ptask, dep_titles in task_pairs:
            ptask.depends_on = [
                title_to_id[t]
                for t in dep_titles
                if t in title_to_id
            ]
            self.task_queue.add_task(ptask)
            added_tasks.append(ptask)

        logger.info(
            f"[PlannerWorkerSwarm] {planner_name} created {len(added_tasks)} tasks"
        )

        # Optionally decompose via sub-planners
        if depth < self.max_planner_depth - 1:
            for ptask in list(added_tasks):
                if ptask.priority == TaskPriority.CRITICAL:
                    self.task_queue.cancel(ptask.id)
                    sub_tasks = self._run_planner(
                        task=f"Decompose this task into smaller subtasks:\n\n{ptask.description}",
                        depth=depth + 1,
                        parent_task_id=ptask.id,
                    )
                    added_tasks.extend(sub_tasks)

        return added_tasks

    def _run_judge(self) -> CycleVerdict:
        """Run the judge agent to evaluate cycle results."""
        schema = BaseTool().base_model_to_dict(CycleVerdict)

        judge = Agent(
            agent_name="CycleJudge",
            agent_description="Evaluates whether the planner-worker cycle achieved the goal",
            system_prompt=JUDGE_SYSTEM_PROMPT,
            model_name=self.judge_model_name,
            max_loops=1,
            base_model=CycleVerdict,
            tools_list_dictionary=[schema],
            output_type="dict-all-except-first",
        )

        all_tasks = self.task_queue.get_all_tasks()
        task_report = "\n".join(
            [
                f"- [{t.status.value}] {t.title}: {t.result or t.error or 'No result'}"
                for t in all_tasks
            ]
        )

        eval_task = (
            f"Original goal: {self._original_task}\n\n"
            f"Task execution report:\n{task_report}\n\n"
            f"Full conversation history:\n{self.conversation.get_str()}\n\n"
            "Evaluate whether the goal has been achieved. "
            "If not, identify specific gaps and provide instructions for the next planning cycle."
        )

        raw_output = judge.run(task=eval_task)

        try:
            verdict = self._parse_structured_output(
                raw_output, CycleVerdict
            )
        except Exception:
            logger.warning(
                "[PlannerWorkerSwarm] Failed to parse judge output, defaulting to incomplete"
            )
            verdict = CycleVerdict(
                is_complete=False,
                overall_quality=0,
                summary="Failed to parse judge evaluation",
                gaps=["Judge output parsing failed"],
                follow_up_instructions="Retry the evaluation",
                needs_fresh_start=False,
            )

        self.conversation.add(
            role="CycleJudge",
            content=f"Quality: {verdict.overall_quality}/10 | Complete: {verdict.is_complete}\n{verdict.summary}",
        )

        return verdict

    def _prepare_next_cycle(self, verdict: CycleVerdict) -> None:
        """Prepare the task queue for the next planner-worker-judge cycle.

        If the judge requested a fresh start, all tasks are discarded to
        combat accumulated drift. Otherwise, only non-terminal tasks are
        cleared so completed results remain available as context.
        """
        if verdict.needs_fresh_start:
            logger.info(
                "[PlannerWorkerSwarm] Judge requested fresh start — clearing all tasks"
            )
            self.task_queue.clear()
        else:
            self.task_queue.clear_non_terminal()

    def get_status(self) -> Dict[str, Any]:
        """Return a structured status report of the swarm and its task queue."""
        return {
            "name": self.name,
            "original_task": self._original_task,
            "queue": self.task_queue.get_status(),
        }

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute the planner-worker-judge cycle.

        Args:
            task: The goal to accomplish.
            img: Optional image input.

        Returns:
            Formatted conversation history per output_type.
        """
        if not task:
            raise ValueError("A task is required")

        self._original_task = task
        self.conversation.add(role="User", content=task)

        verdict = None

        for cycle in range(self.max_loops):
            logger.info(
                f"[PlannerWorkerSwarm] Cycle {cycle + 1}/{self.max_loops}"
            )

            # Between cycles: prepare queue based on judge feedback
            if cycle > 0 and verdict is not None:
                self._prepare_next_cycle(verdict)

            # Phase 1: Planning
            if cycle == 0:
                planner_task = task
            else:
                planner_task = (
                    f"Original goal: {task}\n\n"
                    f"Previous cycle feedback: {verdict.follow_up_instructions}\n"
                    f"Gaps identified: {verdict.gaps}\n\n"
                    "Create new tasks to address these gaps."
                )

            self._run_planner(planner_task)

            # Phase 2: Worker execution
            worker_pool = WorkerPool(
                agents=self.agents,
                task_queue=self.task_queue,
                conversation=self.conversation,
                max_workers=self.max_workers,
                task_timeout=self.task_timeout,
            )
            worker_pool.run(timeout=self.worker_timeout)

            # Log progress
            status = self.task_queue.get_status()
            logger.info(
                f"[PlannerWorkerSwarm] Worker phase done. "
                f"Progress: {status['progress']}, "
                f"Status: {status['status_counts']}"
            )

            # Phase 3: Judge evaluation
            verdict = self._run_judge()

            logger.info(
                f"[PlannerWorkerSwarm] Cycle {cycle + 1} done. "
                f"Quality: {verdict.overall_quality}/10, Complete: {verdict.is_complete}"
            )

            if verdict.is_complete:
                logger.info(
                    f"[PlannerWorkerSwarm] Goal achieved in cycle {cycle + 1}"
                )
                break

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )
