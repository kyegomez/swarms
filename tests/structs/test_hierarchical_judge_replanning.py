import json
from litellm import ModelResponse

from swarms.structs.hiearchical_swarm import HierarchicalSwarm


class ScriptedWorker:
    def __init__(self, name, outputs):
        self.agent_name = name
        self.description = name
        self.system_prompt = name
        self.calls = []
        self.outputs = iter(outputs)

    def run(self, task, **kwargs):
        self.calls.append(task)
        return next(self.outputs)


def test_default_judge_replans_and_reuses_approved_work(monkeypatch):
    """Drive the real director and default judge through LLM responses."""
    reports = []
    requests = []
    planned_orders = [
        {"agent_name": "Research", "task": "Gather facts"},
        {"agent_name": "Writer", "task": "Write report"},
    ]
    responses = iter(
        [
            ("OrderBatch", {"orders": planned_orders}),
            (
                "JudgeReport",
                {
                    "overall_quality": 6,
                    "scores": [],
                    "summary": "The report needs corrected citations.",
                    "verdict": "REVISE",
                    "failed_subtasks": ["Write report"],
                },
            ),
            (
                "OrderBatch",
                {
                    "orders": [
                        planned_orders[0],
                        {
                            "agent_name": "Editor",
                            "task": "Write report",
                        },
                        {
                            "agent_name": "Writer",
                            "task": "Add appendix",
                        },
                    ]
                },
            ),
            (
                "JudgeReport",
                {
                    "overall_quality": 9,
                    "scores": [],
                    "summary": "The revised report is complete.",
                    "verdict": "ACCEPT",
                    "failed_subtasks": [],
                },
            ),
        ]
    )

    def complete(**kwargs):
        requests.append(kwargs)
        name, arguments = next(responses)
        if name == "JudgeReport":
            reports.append(kwargs["tools"])
        return ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_replan",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(
                                        arguments
                                    ),
                                },
                            }
                        ],
                    },
                }
            ]
        )

    monkeypatch.setattr(
        "swarms.utils.litellm_wrapper.completion", complete
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    research = ScriptedWorker(
        "Research", ["verified facts", "duplicate"]
    )
    writer = ScriptedWorker("Writer", ["draft report", "appendix"])
    editor = ScriptedWorker("Editor", ["corrected report"])
    swarm = HierarchicalSwarm(
        agents=[research, writer, editor],
        agent_as_judge=True,
        max_loops=2,
        parallel_execution=False,
        print_on=False,
        add_collaboration_prompt=False,
    )

    swarm.run("Prepare a sourced report")

    assert research.calls == ["Gather facts"]
    assert writer.calls == ["Write report", "Add appendix"]
    assert editor.calls == ["Write report"]
    assert len(requests) == 4
    replanning_prompt = json.dumps(requests[2]["messages"])
    assert "REPLAN REQUIRED" in replanning_prompt
    assert "verified facts" in replanning_prompt
    assert "corrected citations" in replanning_prompt
    assert "verdict" in json.dumps(reports[0])
    assert "failed_subtasks" in json.dumps(reports[0])
