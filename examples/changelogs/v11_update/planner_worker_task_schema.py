from swarms.schemas.planner_worker_schemas import (
    PlannerTask,
    PlannerTaskStatus,
    TaskPriority,
)

# Tasks are created by the Planner automatically, but you can inspect them:
task = PlannerTask(
    title="Gather raw data",
    description="Collect EV market data from public sources",
    priority=TaskPriority.HIGH,
    depends_on=[],  # no dependencies — runs first
    status=PlannerTaskStatus.PENDING,
)
print(task)
