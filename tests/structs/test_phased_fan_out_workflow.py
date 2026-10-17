import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from swarms.structs import phased_fan_out_workflow as dw
from swarms.structs.phased_fan_out_workflow import (
    PhasedFanOutWorkflow,
    PhasedFanOutWorkflowError,
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
        return json.dumps(self.plans.pop(0))

    def __deepcopy__(self, memo):
        return self


class CopiedLead(FakeLead):
    """Lead whose copies are separate objects, each kept in copies."""

    def __init__(self, plans, answer="FINAL"):
        super().__init__(plans, answer)
        self.copies = []

    def __deepcopy__(self, memo):
        copy = type(self)(self.plans, self.answer)
        self.copies.append(copy)
        return copy


@pytest.fixture(autouse=True)
def fake_agents(monkeypatch):
    monkeypatch.setattr(dw, "Agent", FakeAgent)
    FakeAgent.instances = []
    FakeAgent.handler = lambda agent, task: f"{agent.agent_name} done"
    yield


def plan(phases, return_steps, inline=("worker",)):
    return {
        "name": "p",
        "description": "d",
        "inline_agents": [
            {"name": n, "system_prompt": f"You are {n}."}
            for n in inline
        ],
        "phases": phases,
        "return_steps": return_steps,
    }


LIST_SCHEMA = {
    "type": "object",
    "properties": {"files": {"type": "array"}},
}


def fan_out_plan():
    return plan(
        [
            {
                "title": "Scan",
                "steps": [
                    {
                        "id": "scan",
                        "agent": "worker",
                        "prompt": "List files in {task}",
                        "output_schema": LIST_SCHEMA,
                    }
                ],
            },
            {
                "title": "Review",
                "steps": [
                    {
                        "id": "review",
                        "agent": "worker",
                        "prompt": "Review {item}",
                        "for_each": "scan.files",
                    }
                ],
            },
        ],
        ["review"],
    )


def file_handler(n):
    def handler(agent, task):
        if task.startswith("List files"):
            return json.dumps(
                {"files": [f"f{i}.py" for i in range(n)]}
            )
        return f"reviewed {task.split()[1]}"

    return handler


def test_phases_run_in_order_and_outputs_flow_forward():
    lead = FakeLead(
        [
            plan(
                [
                    {
                        "title": "A",
                        "steps": [
                            {
                                "id": "a",
                                "agent": "worker",
                                "prompt": "A {task}",
                            }
                        ],
                    },
                    {
                        "title": "B",
                        "steps": [
                            {
                                "id": "b",
                                "agent": "worker",
                                "prompt": "use {a}",
                            }
                        ],
                    },
                ],
                ["b"],
            )
        ]
    )
    FakeAgent.handler = lambda agent, task: f"out<{task}>"
    workflow = PhasedFanOutWorkflow(lead_agent=lead)

    assert workflow.run("task") == "FINAL"
    assert workflow.last_run["outputs"] == {
        "a": "out<A task>",
        "b": "out<use out<A task>>",
    }
    assert "out<use out<A task>>" in lead.prompts[-1]


def test_steps_in_a_phase_run_in_parallel():
    steps = [
        {
            "id": f"s{i}",
            "agent": "worker",
            "prompt": f"wait {i} {{task}}",
        }
        for i in range(4)
    ]
    lead = FakeLead([plan([{"title": "P", "steps": steps}], ["s0"])])
    barrier = threading.Barrier(4, timeout=5)

    def handler(agent, task):
        barrier.wait()
        return "ok"

    FakeAgent.handler = handler
    PhasedFanOutWorkflow(lead_agent=lead).run("task")
    assert not barrier.broken


def test_max_concurrent_agents_bounds_parallelism():
    steps = [
        {
            "id": f"s{i}",
            "agent": "worker",
            "prompt": f"x {i} {{task}}",
        }
        for i in range(6)
    ]
    lead = FakeLead([plan([{"title": "P", "steps": steps}], ["s0"])])
    lock = threading.Lock()
    active = {"now": 0, "peak": 0}

    def handler(agent, task):
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        time.sleep(0.05)
        with lock:
            active["now"] -= 1
        return "ok"

    FakeAgent.handler = handler
    PhasedFanOutWorkflow(
        lead_agent=lead, max_concurrent_agents=2
    ).run("task")
    assert active["peak"] == 2


def test_fan_out_width_comes_from_runtime_output():
    lead = FakeLead([fan_out_plan()])
    FakeAgent.handler = file_handler(7)
    workflow = PhasedFanOutWorkflow(lead_agent=lead)
    workflow.run("task")

    assert workflow.last_run["outputs"]["review"] == [
        f"reviewed f{i}.py" for i in range(7)
    ]
    assert workflow.last_run["agent_count"] == 8


def test_json_followed_by_prose_is_parsed():
    def handler(agent, task):
        if task.startswith("List files"):
            return (
                "Files found:\n"
                + json.dumps({"files": ["a.py", "b.py"]})
                + "\nLet me know if you need more."
            )
        return f"reviewed {task.split()[1]}"

    FakeAgent.handler = handler
    workflow = PhasedFanOutWorkflow(
        lead_agent=FakeLead([fan_out_plan()])
    )
    workflow.run("repo")

    assert workflow.last_run["outputs"]["review"] == [
        "reviewed a.py",
        "reviewed b.py",
    ]
    with pytest.raises(json.JSONDecodeError):
        dw._parse_json('{"files": ["a.py"],} Thanks!')
    with pytest.raises(json.JSONDecodeError):
        dw._parse_json(
            "I reviewed [3] files and found nothing wrong."
        )


def test_fan_out_over_max_agents_stops_before_starting():
    lead = FakeLead([fan_out_plan()])
    FakeAgent.handler = file_handler(5)
    workflow = PhasedFanOutWorkflow(lead_agent=lead, max_agents=4)

    with pytest.raises(PhasedFanOutWorkflowError, match="max_agents"):
        workflow.run("task")
    assert len(FakeAgent.instances) == 1


def test_failed_fan_out_agent_leaves_none_and_run_continues():
    lead = FakeLead([fan_out_plan()])
    base = file_handler(3)

    def handler(agent, task):
        if "f1.py" in task:
            raise RuntimeError("boom")
        return base(agent, task)

    FakeAgent.handler = handler
    workflow = PhasedFanOutWorkflow(lead_agent=lead, max_retries=1)

    assert workflow.run("task") == "FINAL"
    assert workflow.last_run["outputs"]["review"] == [
        "reviewed f0.py",
        None,
        "reviewed f2.py",
    ]
    assert workflow.last_run["failures"][0]["item"] == "f1.py"
    attempts = [
        a for a in FakeAgent.instances if "f1.py" in str(a.history)
    ]
    assert len(attempts) == 2


def test_failed_agent_retries_then_succeeds():
    lead = FakeLead(
        [
            plan(
                [
                    {
                        "title": "P",
                        "steps": [
                            {
                                "id": "a",
                                "agent": "worker",
                                "prompt": "x {task}",
                            }
                        ],
                    }
                ],
                ["a"],
            )
        ]
    )
    calls = {"n": 0}

    def handler(agent, task):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("flaky")
        return "ok"

    FakeAgent.handler = handler
    workflow = PhasedFanOutWorkflow(lead_agent=lead, max_retries=1)
    workflow.run("task")
    assert workflow.last_run["outputs"]["a"] == "ok"
    assert workflow.last_run["failures"] == []


def test_failed_single_step_raises():
    lead = FakeLead(
        [
            plan(
                [
                    {
                        "title": "P",
                        "steps": [
                            {
                                "id": "a",
                                "agent": "worker",
                                "prompt": "x {task}",
                            }
                        ],
                    }
                ],
                ["a"],
            )
        ]
    )

    def handler(agent, task):
        raise RuntimeError("down")

    FakeAgent.handler = handler
    with pytest.raises(PhasedFanOutWorkflowError, match="Step 'a'"):
        PhasedFanOutWorkflow(lead_agent=lead, max_retries=2).run(
            "task"
        )
    assert len(FakeAgent.instances) == 3


def test_invalid_plan_is_repaired_and_never_executed():
    bad = plan(
        [
            {
                "title": "P",
                "steps": [
                    {
                        "id": "a",
                        "agent": "ghost",
                        "prompt": "use {b}",
                    },
                    {
                        "id": "b",
                        "agent": "worker",
                        "prompt": "x {task}",
                    },
                ],
            }
        ],
        ["a"],
    )
    good = plan(
        [
            {
                "title": "P",
                "steps": [
                    {
                        "id": "a",
                        "agent": "worker",
                        "prompt": "x {task}",
                    }
                ],
            }
        ],
        ["a"],
    )
    lead = FakeLead(["not json at all", bad, good])
    workflow = PhasedFanOutWorkflow(
        lead_agent=lead, max_plan_attempts=3
    )
    workflow.run("task")

    assert len(FakeAgent.instances) == 1
    assert "unknown agent 'ghost'" in lead.prompts[2]
    assert "not in an earlier phase" in lead.prompts[2]
    assert workflow.last_plan.phases[0].steps[0].agent == "worker"


def test_plan_attempts_run_out():
    bad = plan(
        [
            {
                "title": "P",
                "steps": [
                    {
                        "id": "a",
                        "agent": "ghost",
                        "prompt": "x {task}",
                    }
                ],
            }
        ],
        ["a"],
    )
    lead = FakeLead([bad, bad])
    with pytest.raises(
        PhasedFanOutWorkflowError, match="No valid plan"
    ):
        PhasedFanOutWorkflow(
            lead_agent=lead, max_plan_attempts=2
        ).run("t")
    assert FakeAgent.instances == []


def test_validate_plan_rules():
    workflow = PhasedFanOutWorkflow(
        lead_agent=FakeLead([]),
        agents=[FakeAgent("analyst", "Analyses things.")],
        allow_inline_agents=False,
        max_agents=1,
    )
    errors = workflow.validate_plan(
        dw.WorkflowPlan.model_validate(
            plan(
                [
                    {
                        "title": "P",
                        "steps": [
                            {
                                "id": "a",
                                "agent": "analyst",
                                "prompt": "x {task}",
                            },
                            {
                                "id": "b",
                                "agent": "worker",
                                "prompt": "{item}",
                            },
                        ],
                    },
                    {
                        "title": "Q",
                        "steps": [
                            {
                                "id": "c",
                                "agent": "analyst",
                                "prompt": "{item}",
                                "for_each": "a.files",
                            }
                        ],
                    },
                ],
                ["missing"],
            )
        )
    )
    text = "\n".join(errors)
    assert "Inline agents are not allowed" in text
    assert "{item} without for_each" in text
    assert "not declared in the output_schema of 'a'" in text
    assert "unknown step 'missing'" in text
    assert "limit is 1" in text


def test_predefined_agents_are_copied_per_use():
    analyst = FakeAgent("analyst", "Analyses things.")
    FakeAgent.instances = []
    one_step = plan(
        [
            {
                "title": "P",
                "steps": [
                    {
                        "id": "a",
                        "agent": "analyst",
                        "prompt": "x {task}",
                    },
                    {
                        "id": "b",
                        "agent": "analyst",
                        "prompt": "y {task}",
                    },
                ],
            }
        ],
        ["a", "b"],
        inline=(),
    )
    lead = FakeLead([one_step, one_step])
    workflow = PhasedFanOutWorkflow(lead_agent=lead, agents=[analyst])
    workflow.run("first")
    workflow.run("second")

    assert analyst.history == []
    assert len(FakeAgent.instances) == 4
    assert all(len(a.history) == 1 for a in FakeAgent.instances)


def test_usage_sums_lead_and_workers():
    lead = FakeLead([fan_out_plan()])
    FakeAgent.handler = file_handler(2)
    workflow = PhasedFanOutWorkflow(lead_agent=lead)
    workflow.run("task")
    assert workflow.usage["input_tokens"] == 100 + 3 * 10
    assert workflow.last_run["usage"]["output_tokens"] == 5 + 3


def test_task_placeholder_inserts_the_original_task():
    lead = FakeLead(
        [
            plan(
                [
                    {
                        "title": "P",
                        "steps": [
                            {
                                "id": "a",
                                "agent": "worker",
                                "prompt": "Read this: {task}",
                            }
                        ],
                    }
                ],
                ["a"],
            )
        ]
    )
    FakeAgent.handler = lambda agent, task: task
    workflow = PhasedFanOutWorkflow(lead_agent=lead)
    workflow.run("CONTRACT TEXT")
    assert (
        workflow.last_run["outputs"]["a"]
        == "Read this: CONTRACT TEXT"
    )


def test_step_without_placeholders_is_rejected():
    workflow = PhasedFanOutWorkflow(lead_agent=FakeLead([]))
    errors = workflow.validate_plan(
        dw.WorkflowPlan.model_validate(
            plan(
                [
                    {
                        "title": "P",
                        "steps": [
                            {
                                "id": "a",
                                "agent": "worker",
                                "prompt": "Read the contracts in the task.",
                            },
                            {
                                "id": "b",
                                "agent": "worker",
                                "prompt": "Read {task}",
                            },
                        ],
                    }
                ],
                ["a"],
            )
        )
    )
    assert any("Step 'a' has no {task}" in e for e in errors)
    assert not any("Step 'b'" in e for e in errors)


def test_identical_steps_in_a_phase_are_rejected():
    workflow = PhasedFanOutWorkflow(lead_agent=FakeLead([]))
    same = {"agent": "worker", "prompt": "Read {task}"}
    errors = workflow.validate_plan(
        dw.WorkflowPlan.model_validate(
            plan(
                [
                    {
                        "title": "P",
                        "steps": [
                            {"id": "a", **same},
                            {"id": "b", **same},
                        ],
                    }
                ],
                ["a", "b"],
            )
        )
    )
    assert any("Steps 'a' and 'b'" in e for e in errors)


def test_plan_before_any_run_uses_one_fresh_copy_of_the_lead():
    lead = CopiedLead(["not json", fan_out_plan()])
    workflow = PhasedFanOutWorkflow(lead_agent=lead)

    assert isinstance(workflow.plan("task"), dw.WorkflowPlan)
    assert lead.prompts == []
    assert [len(copy.prompts) for copy in lead.copies] == [2]


def test_plan_after_a_run_uses_a_fresh_copy_of_the_lead():
    lead = CopiedLead([fan_out_plan()] * 3)
    base = file_handler(2)

    def handler(agent, task):
        if "f1.py" in task:
            raise RuntimeError("boom")
        return base(agent, task)

    FakeAgent.handler = handler
    workflow = PhasedFanOutWorkflow(lead_agent=lead, max_retries=0)
    workflow.run("run task")
    last_run, last_plan = workflow.last_run, workflow.last_plan
    assert len(last_run["failures"]) == 1
    # repr records nested contents and, through default reprs, agent identities.
    saved_run, saved_plan = repr(last_run), last_plan.model_dump()
    usage = workflow.usage

    for _ in range(2):
        assert isinstance(workflow.plan("plan task"), dw.WorkflowPlan)
    assert [len(copy.prompts) for copy in lead.copies] == [2, 1, 1]
    # Later copies get only invalid plans, so this plan() fails.
    lead.plans = ["not json"] * 3
    with pytest.raises(
        PhasedFanOutWorkflowError, match="No valid plan"
    ):
        workflow.plan("plan task")
    assert workflow.last_run is last_run
    assert repr(workflow.last_run) == saved_run
    assert workflow.last_plan is last_plan
    assert workflow.last_plan.model_dump() == saved_plan
    assert workflow.usage == usage


def test_overlapping_runs_plan_and_combine_with_their_own_lead():
    a_planning, b_planned = threading.Event(), threading.Event()
    overlapped = []

    class Lead(CopiedLead):
        def run(self, task, img=None):
            if "Outputs:" not in task:
                if task.startswith("Task:\nA\n"):
                    a_planning.set()
                    overlapped.append(b_planned.wait(5))
                else:
                    b_planned.set()
            return super().run(task, img)

    one_step = plan(
        [
            {
                "title": "P",
                "steps": [
                    {
                        "id": "a",
                        "agent": "worker",
                        "prompt": "x {task}",
                    }
                ],
            }
        ],
        ["a"],
    )
    lead = Lead([one_step])
    workflow = PhasedFanOutWorkflow(lead_agent=lead)
    with ThreadPoolExecutor(max_workers=2) as pool:
        run_a = pool.submit(workflow.run, "A")
        assert a_planning.wait(5)
        run_b = pool.submit(workflow.run, "B")
        assert [run_a.result(), run_b.result()] == ["FINAL", "FINAL"]

    assert overlapped == [True]
    assert lead.prompts == []
    # Every lead prompt starts "Task:\n<task>\n".
    tasks_seen = sorted(
        [prompt.split("\n")[1] for prompt in copy.prompts]
        for copy in lead.copies
    )
    assert tasks_seen == [["A", "A"], ["B", "B"]]
