import time

from swarms.planners.task_queue import InMemoryTaskQueue, Task
from swarms.planners.sub_planner import SubPlanner


def test_inmemory_queue_push_pop_ack():
    q = InMemoryTaskQueue()
    t = Task(id="t1", title="Top-level task", description="Do X")
    q.push_task(t)
    popped = q.pop_task_atomic()
    assert popped is not None
    assert popped.id == "t1"
    assert popped.status == "in-progress"
    q.ack_task(popped.id, success=True)
    got = q.get_task(popped.id)
    assert got is not None
    assert got.status == "completed"


def test_subplanner_breakdown_and_enqueue():
    q = InMemoryTaskQueue()
    sub = SubPlanner(task_queue=q)
    parent = Task(id="p1", title="Parent", description="Big goal")
    subs = sub.breakdown_task(parent)
    assert len(subs) == 2
    sub.plan_and_enqueue(parent)
    # After enqueue, list pending should contain two entries
    pending = q.list_pending()
    assert len(pending) == 2
