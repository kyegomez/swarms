import gc
import json
import weakref

import pytest

from swarms.structs import phased_fan_out_workflow as dw
from swarms.structs.phased_fan_out_workflow import (
    PhasedFanOutWorkflow,
    _shutdown_executor,
)


class DummyAgent:
    def __init__(self, agent_name="agent-1", **kwargs):
        self.agent_name = agent_name
        self.agent_description = "A dummy agent for testing"
        self.system_prompt = "You are a test agent"
        self.output_type = "final"
        self.history = []
        self.usage = {"input_tokens": 5, "output_tokens": 5}

    def run(self, task, img=None):
        return f"processed: {task}"

    def __deepcopy__(self, memo):
        return DummyAgent(self.agent_name)


class FakeLead:
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
        return json.dumps(
            {
                "name": "p",
                "description": "d",
                "inline_agents": [{"name": "worker", "system_prompt": "prompt"}],
                "phases": [
                    {
                        "title": "Phase 1",
                        "steps": [{"id": "s1", "agent": "worker", "prompt": "{task}"}],
                    }
                ],
                "return_steps": ["s1"],
            }
        )

    def __deepcopy__(self, memo):
        return self


def test_executor_is_none_initially():
    agent = DummyAgent("agent-1")
    wf = PhasedFanOutWorkflow(
        lead_agent=agent,
        agents=[agent],
        allow_inline_agents=False,
    )
    assert getattr(wf, "_executor", "NOT_FOUND") is None
    assert getattr(wf, "_finalizer", "NOT_FOUND") is None


def test_get_executor_reuses_instance():
    agent = DummyAgent("agent-1")
    wf = PhasedFanOutWorkflow(
        lead_agent=agent,
        agents=[agent],
        allow_inline_agents=False,
        max_concurrent_agents=4,
    )
    pool1 = wf._get_executor()
    pool2 = wf._get_executor()
    assert pool1 is pool2
    assert wf._executor is pool1
    wf.close()
    assert wf._executor is None


def test_get_executor_dynamic_upscaling():
    agent = DummyAgent("agent-1")
    wf = PhasedFanOutWorkflow(
        lead_agent=agent,
        agents=[agent],
        allow_inline_agents=False,
        max_concurrent_agents=2,
    )
    pool1 = wf._get_executor(2)
    assert pool1._max_workers == 2

    # Upscaling to 5 workers should retire pool1 and create pool2
    pool2 = wf._get_executor(5)
    assert pool2._max_workers == 5
    assert pool2 is not pool1
    assert pool1._shutdown is True

    wf.close()


def test_close_teardown():
    agent = DummyAgent("agent-1")
    wf = PhasedFanOutWorkflow(
        lead_agent=agent,
        agents=[agent],
        allow_inline_agents=False,
    )
    pool = wf._get_executor()
    assert pool._shutdown is False
    wf.close()
    assert wf._executor is None
    assert wf._finalizer is None
    assert pool._shutdown is True


def test_context_manager():
    agent = DummyAgent("agent-1")
    with PhasedFanOutWorkflow(
        lead_agent=agent,
        agents=[agent],
        allow_inline_agents=False,
    ) as wf:
        pool = wf._get_executor()
        assert pool._shutdown is False
    assert wf._executor is None
    assert pool._shutdown is True


def test_weakref_finalizer_cleanup():
    agent = DummyAgent("agent-1")
    wf = PhasedFanOutWorkflow(
        lead_agent=agent,
        agents=[agent],
        allow_inline_agents=False,
    )
    pool = wf._get_executor()
    pool_ref = weakref.ref(pool)
    assert pool._shutdown is False

    del wf
    gc.collect()

    # The executor should have been shut down by the finalizer
    assert pool_ref() is not None
    assert pool_ref()._shutdown is True
    _shutdown_executor(pool_ref())


def test_executor_reused_during_run_phases(monkeypatch):
    monkeypatch.setattr(dw, "Agent", DummyAgent)
    plan_dict = {
        "name": "p",
        "description": "d",
        "inline_agents": [{"name": "worker", "system_prompt": "prompt"}],
        "phases": [
            {
                "title": "Phase 1",
                "steps": [{"id": "s1", "agent": "worker", "prompt": "{task}"}],
            },
            {
                "title": "Phase 2",
                "steps": [{"id": "s2", "agent": "worker", "prompt": "{s1}"}],
            },
        ],
        "return_steps": ["s1", "s2"],
    }
    lead = FakeLead([plan_dict, plan_dict])

    wf = PhasedFanOutWorkflow(
        lead_agent=lead,
        max_concurrent_agents=2,
    )

    result = wf.run("Do work")
    assert result == "FINAL"
    # Verify executor was created and retained across the phases
    assert wf._executor is not None
    pool = wf._executor

    # Run again: executor should be reused
    result2 = wf.run("Do more work")
    assert result2 == "FINAL"
    assert wf._executor is pool
    assert pool._shutdown is False

    wf.close()
    assert pool._shutdown is True
    assert wf._executor is None
