import time
import uuid
from enum import Enum, IntEnum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TaskPriority(IntEnum):
    """Priority levels for tasks in the queue."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class PlannerTaskStatus(str, Enum):
    """Status of a task in the planner-worker queue.

    Transitions:
        PENDING -> CLAIMED -> RUNNING -> COMPLETED
        PENDING -> CLAIMED -> RUNNING -> FAILED -> PENDING (retry)
        Any non-terminal -> CANCELLED
    """

    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlannerTask(BaseModel):
    """A single task in the shared planner-worker task queue.

    Created by planner agents, consumed by worker agents.
    The `version` field enables optimistic concurrency control.
    """

    id: str = Field(
        default_factory=lambda: f"ptask-{uuid.uuid4().hex[:10]}",
        description="Unique task identifier",
    )
    title: str = Field(
        ...,
        description="Short, descriptive title of the task",
    )
    description: str = Field(
        ...,
        description="Detailed description of what needs to be done",
    )
    priority: TaskPriority = Field(
        default=TaskPriority.NORMAL,
        description="Task priority level",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of task IDs that must complete before this task can start",
    )
    parent_task_id: Optional[str] = Field(
        default=None,
        description="ID of the parent task if decomposed from a larger task",
    )
    status: PlannerTaskStatus = Field(
        default=PlannerTaskStatus.PENDING,
        description="Current task status",
    )
    assigned_worker: Optional[str] = Field(
        default=None,
        description="Name of the worker agent that claimed this task",
    )
    result: Optional[str] = Field(
        default=None,
        description="Result of task execution",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if task failed",
    )
    retries: int = Field(
        default=0,
        description="Number of retry attempts so far",
    )
    max_retries: int = Field(
        default=2,
        description="Maximum retry attempts before permanent failure",
    )
    version: int = Field(
        default=0,
        description="Optimistic concurrency version counter",
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp of task creation",
    )
    completed_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp of task completion",
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Arbitrary metadata",
    )


class PlannerTaskOutput(BaseModel):
    """A single task definition as output from a planner agent."""

    title: str = Field(
        ...,
        description="Short, descriptive title",
    )
    description: str = Field(
        ...,
        description="Detailed description of what a worker agent should do",
    )
    priority: int = Field(
        default=1,
        description="Priority: 0=LOW, 1=NORMAL, 2=HIGH, 3=CRITICAL",
    )
    depends_on_titles: List[str] = Field(
        default_factory=list,
        description="Titles of other tasks in this plan that must complete first",
    )


class PlannerTaskSpec(BaseModel):
    """Structured output from a planner agent.

    The planner produces a plan narrative and a list of concrete tasks.
    """

    plan: str = Field(
        ...,
        description="Narrative explanation of the plan: what needs to be done, in what order, and why.",
    )
    tasks: List[PlannerTaskOutput] = Field(
        ...,
        description="List of concrete tasks to add to the queue.",
    )


class CycleVerdict(BaseModel):
    """Structured output from the judge agent after evaluating a cycle."""

    is_complete: bool = Field(
        ...,
        description="True if the overall goal has been satisfactorily achieved",
    )
    overall_quality: int = Field(
        ...,
        ge=0,
        le=10,
        description="Quality score 0-10 of the combined results",
    )
    summary: str = Field(
        ...,
        description="Summary assessment of the cycle results",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="Specific gaps or issues that need addressing in a follow-up cycle",
    )
    follow_up_instructions: Optional[str] = Field(
        default=None,
        description="Instructions for the planner if another cycle is needed",
    )
    needs_fresh_start: bool = Field(
        default=False,
        description=(
            "True if accumulated drift or systemic issues require a complete "
            "restart rather than incremental gap-filling. When True, all prior "
            "tasks are discarded and the planner begins from scratch with the "
            "original goal plus judge feedback."
        ),
    )
