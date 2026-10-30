import threading
import pytest
from swarms.structs.async_subagent import SubagentRegistry
from swarms.structs.autonomous_loop_utils import assign_task_tool


class EventStubAgent:
    def __init__(self, name, event=None, result="done"):
        self.agent_name = name
        self.event = event
        self.result = result

    def run(self, task):
        if self.event:
            self.event.wait(timeout=5)
        return f"{self.agent_name}: {self.result}"


def test_gather_filters_by_task_ids():
    """gather(task_ids=[...]) must only wait on and return results for specified tasks."""
    reg = SubagentRegistry()
    event_block = threading.Event()
    slow_agent = EventStubAgent("slow", event=event_block, result="slow_res")
    fast_agent = EventStubAgent("fast", event=None, result="fast_res")

    slow_id = reg.spawn(slow_agent, "slow task")
    fast_id = reg.spawn(fast_agent, "fast task")

    # gather only fast_id: must complete immediately without waiting on event_block
    results = reg.gather(task_ids=[fast_id], timeout=1.0)
    assert results == ["fast: fast_res"]

    # Now unblock slow task and clean up
    event_block.set()
    reg.gather(task_ids=[slow_id])
    reg.shutdown()


def test_gather_empty_task_ids_returns_empty():
    """gather(task_ids=[]) must return [] immediately."""
    reg = SubagentRegistry()
    assert reg.gather(task_ids=[]) == []
    reg.shutdown()


def test_assign_task_blocking_waits_only_on_dispatched_tasks():
    """assign_task_tool with wait_for_completion=True must not wait on prior background tasks."""
    class DummyMemory:
        def add(self, role, content):
            pass

    event_block = threading.Event()
    bg_agent = EventStubAgent("bg", event=event_block, result="bg_res")
    sync_agent = EventStubAgent("sync", event=None, result="sync_res")

    class Parent:
        id = "parent-agent"
        verbose = False
        max_subagent_depth = 3
        short_memory = DummyMemory()
        sub_agents = {
            "bg": {"agent": bg_agent, "name": "bg"},
            "sync": {"agent": sync_agent, "name": "sync"},
        }

    parent = Parent()

    # 1. Start long background task (fire and forget)
    assign_task_tool(
        parent,
        [{"agent_id": "bg", "task": "long background"}],
        wait_for_completion=False,
    )

    # 2. Run blocking assign_task for fast agent: must complete immediately while bg is still blocked
    output = assign_task_tool(
        parent,
        [{"agent_id": "sync", "task": "quick sync"}],
        wait_for_completion=True,
    )

    assert "Completed 1 task assignment(s)" in output
    assert "sync: sync_res" in output

    # Clean up background task
    event_block.set()
    if hasattr(parent, "_subagent_registry") and parent._subagent_registry:
        parent._subagent_registry.shutdown()
