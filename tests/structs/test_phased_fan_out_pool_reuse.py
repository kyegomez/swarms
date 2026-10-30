import gc
import json
import pytest

from swarms.structs import phased_fan_out_workflow as dw
from swarms.structs.phased_fan_out_workflow import (
    PhasedFanOutWorkflow,
)


class FakeAgent:
    """Offline stand-in for Agent whose replies come from a handler."""

    handler = None
    instances = []

    def __init__(
        self,
        agent_name="fake",
        agent_description="",
        system_prompt="",
        handler=None,
        **kwargs,
    ):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.system_prompt = system_prompt
        self.output_type = kwargs.get("output_type")
        self.handler = handler or FakeAgent.handler
        self.history = []
        self.usage = {"input_tokens": 10, "output_tokens": 1}
        FakeAgent.instances.append(self)

    def run(self, task, img=None):
        self.history.append(task)
        return self.handler(self, task)

    def __deepcopy__(self, memo):
        return FakeAgent(
            self.agent_name,
            self.agent_description,
            self.system_prompt,
            handler=self.handler,
        )


class FakeLead:
    """Lead whose plans and final answer are scripted."""

    def __init__(self, plans, answer="FINAL"):
        self.agent_name = "lead"
        self.model_name = "fake-model"
        self.output_type = "final"
        self.plans = list(plans)
        self.answer = answer
        self.prompts = []
        self.usage = {"input_tokens": 100, "output_tokens": 5}

    def run(self, task, img=None):
        self.prompts.append(task)
        if task.startswith("Task:") and "Outputs:" in task:
            return self.answer
        if self.plans:
            return json.dumps(self.plans.pop(0))
        return json.dumps(sample_plan())

    def __deepcopy__(self, memo):
        return self


@pytest.fixture(autouse=True)
def fake_agents(monkeypatch):
    monkeypatch.setattr(dw, "Agent", FakeAgent)
    FakeAgent.instances = []
    FakeAgent.handler = lambda agent, task: f"{agent.agent_name} done"
    yield


def sample_plan():
    return {
        "name": "two_phase_plan",
        "description": "plan with two phases",
        "inline_agents": [
            {"name": "worker_a", "system_prompt": "worker a"},
            {"name": "worker_b", "system_prompt": "worker b"},
        ],
        "phases": [
            {
                "title": "Phase 1",
                "steps": [
                    {"id": "p1_s1", "agent": "worker_a", "prompt": "Process: {task}"},
                ],
            },
            {
                "title": "Phase 2",
                "steps": [
                    {"id": "p2_s1", "agent": "worker_b", "prompt": "Combine: {p1_s1}"},
                ],
            },
        ],
        "return_steps": ["p1_s1", "p2_s1"],
    }


def test_executor_reused_across_runs_and_phases():
    """Verify that multiple run() calls and phases reuse the same thread pool executor."""
    lead = FakeLead([sample_plan(), sample_plan()])
    workflow = PhasedFanOutWorkflow(
        name="test_reuse",
        lead_agent=lead,
        max_concurrent_agents=4,
    )

    assert getattr(workflow, "_executor", None) is None
    workflow.run("Task 1")
    first_executor = workflow._executor
    assert first_executor is not None
    assert not first_executor._shutdown

    workflow.run("Task 2")
    second_executor = workflow._executor
    assert second_executor is first_executor
    assert not second_executor._shutdown

    workflow.close()
    assert workflow._executor is None
    assert first_executor._shutdown


def test_executor_rebuilt_when_max_concurrent_agents_changes():
    """Rebuilding executor when max_concurrent_agents changes."""
    lead = FakeLead([sample_plan(), sample_plan()])
    workflow = PhasedFanOutWorkflow(
        name="test_resize",
        lead_agent=lead,
        max_concurrent_agents=4,
    )

    workflow.run("Task 1")
    first_executor = workflow._executor
    assert first_executor._max_workers == 4

    workflow.max_concurrent_agents = 8
    workflow.run("Task 2")
    second_executor = workflow._executor
    assert second_executor is not first_executor
    assert second_executor._max_workers == 8
    assert first_executor._shutdown

    workflow.close()


def test_executor_context_manager():
    """Context manager properly cleans up executor on exit."""
    lead = FakeLead([sample_plan()])
    with PhasedFanOutWorkflow(
        name="test_context",
        lead_agent=lead,
        max_concurrent_agents=4,
    ) as workflow:
        workflow.run("Task 1")
        executor = workflow._executor
        assert executor is not None
        assert not executor._shutdown

    assert workflow._executor is None
    assert executor._shutdown


def test_weakref_finalizer_shuts_down_executor():
    """Weakref finalizer shuts down executor when workflow is garbage collected."""
    def _create_and_run():
        lead = FakeLead([sample_plan()])
        wf = PhasedFanOutWorkflow(
            name="test_gc",
            lead_agent=lead,
            max_concurrent_agents=4,
        )
        wf.run("Task 1")
        return wf._executor

    executor = _create_and_run()
    gc.collect()
    assert executor._shutdown


def test_nested_workflow_execution_no_deadlock():
    """Nested workflow runs inside an agent task without deadlocking because pools are per-instance."""
    def nested_handler(agent, task):
        if agent.agent_name == "outer_worker":
            inner_lead = FakeLead([sample_plan()])
            inner_wf = PhasedFanOutWorkflow(
                name="inner_wf",
                lead_agent=inner_lead,
                max_concurrent_agents=2,
            )
            res = inner_wf.run("Inner task")
            inner_wf.close()
            return f"inner result: {res}"
        return f"{agent.agent_name} done"

    FakeAgent.handler = nested_handler

    outer_plan = {
        "name": "outer_plan",
        "description": "outer plan",
        "inline_agents": [
            {"name": "outer_worker", "system_prompt": "outer"},
        ],
        "phases": [
            {
                "title": "Outer Phase 1",
                "steps": [
                    {"id": "out1", "agent": "outer_worker", "prompt": "Process: {task}"},
                ],
            }
        ],
        "return_steps": ["out1"],
    }

    outer_lead = FakeLead([outer_plan])
    outer_wf = PhasedFanOutWorkflow(
        name="outer_wf",
        lead_agent=outer_lead,
        max_concurrent_agents=2,
    )
    result = outer_wf.run("Outer task")
    assert outer_wf._executor is not None
    assert not outer_wf._executor._shutdown
    outer_wf.close()
    assert outer_wf._executor is None
