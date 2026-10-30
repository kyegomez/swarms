import json
import threading
import pytest
from swarms.structs.phased_fan_out_workflow import PhasedFanOutWorkflow


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
        return json.dumps(self.plans.pop(0))

    def __deepcopy__(self, memo):
        return self


class FakeWorker:
    def __init__(self, agent_name="worker"):
        self.agent_name = agent_name
        self.agent_description = "Worker agent"
        self.usage = {"input_tokens": 10, "output_tokens": 1}

    def run(self, task, img=None):
        return f"result: {task}"

    def __deepcopy__(self, memo):
        return self


def test_plan_before_run_succeeds():
    """plan() must succeed on a workflow that has not been run yet."""
    plan_dict = {
        "name": "test_plan",
        "description": "plan description",
        "phases": [{"title": "Phase1", "steps": [{"id": "s1", "agent": "worker", "prompt": "{task}"}]}],
        "return_steps": ["s1"],
    }
    lead = FakeLead(plans=[plan_dict])
    worker = FakeWorker("worker")
    workflow = PhasedFanOutWorkflow(lead_agent=lead, agents=[worker], allow_inline_agents=False)
    plan = workflow.plan("Summarise three contracts")
    assert plan is not None
    assert plan.name == "test_plan"


def test_concurrent_runs_isolate_leads():
    """Two concurrent runs on the same workflow instance must each combine with their own lead."""
    PLAN = json.dumps({
        "name": "p",
        "description": "d",
        "phases": [{"title": "P", "steps": [{"id": "s", "agent": "worker", "prompt": "{task}"}]}],
        "return_steps": ["s"],
    })
    a_planning, b_planned = threading.Event(), threading.Event()
    overlapped = []

    class Worker:
        agent_name, agent_description, usage = "worker", "echoes the task", {}
        def run(self, task, img=None):
            return task

    class Lead:
        agent_name, model_name, usage = "lead", "fake", {}
        def __init__(self):
            self.task = None

        def __deepcopy__(self, memo):
            copy_obj = Lead()
            copy_obj.task = self.task
            return copy_obj

        def run(self, prompt, img=None):
            if "Outputs:" in prompt:
                return f"combined by the lead that planned task {self.task}"
            lines = prompt.split("\n")
            task_val = "unknown"
            for i, line in enumerate(lines):
                if line.strip() == "Task:":
                    task_val = lines[i + 1].strip()
                    break
            self.task = task_val
            if self.task == "A":
                a_planning.set()
                overlapped.append(b_planned.wait(5))
            else:
                b_planned.set()
            return PLAN

    workflow = PhasedFanOutWorkflow(lead_agent=Lead(), agents=[Worker()], allow_inline_agents=False)
    results = {}
    run_a = threading.Thread(target=lambda: results.update(A=workflow.run("A")))
    run_a.start()
    assert a_planning.wait(5), "run A never started planning"
    run_b = threading.Thread(target=lambda: results.update(B=workflow.run("B")))
    run_b.start()
    run_a.join()
    run_b.join()
    assert overlapped == [True], "the runs did not overlap"
    assert "combined by the lead that planned task A" in str(results["A"])
    assert "combined by the lead that planned task B" in str(results["B"])
