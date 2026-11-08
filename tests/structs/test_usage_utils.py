import json
import types

from swarms import (
    Agent,
    ConcurrentWorkflow,
    SequentialWorkflow,
    TreeOfThoughts,
)
from swarms.structs.phased_fan_out_workflow import (
    PhasedFanOutWorkflow,
)
from swarms.structs.usage_utils import total_usage
from swarms.utils.litellm_wrapper import LiteLLM, empty_usage


def _usage_agent(name, input_tokens, output_tokens):
    """An offline agent whose run reports a fixed usage through the real hook."""
    agent = Agent(agent_name=name, max_loops=1, print_on=False)

    def run(self, task=None, *args, **kwargs):
        self._add_usage(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        )
        return f"{name}:{task}"

    agent.run = types.MethodType(run, agent)
    return agent


def test_total_usage_counts_each_agent_client_and_counter_once():
    """Shared agents, cycles and repeated counters are each counted once."""
    a, b = _usage_agent("A", 100, 10), _usage_agent("B", 50, 5)
    a.run("t")
    b.run("t")
    workflow = ConcurrentWorkflow(agents=[a, b])
    workflow.itself = workflow
    client = LiteLLM(model_name="gpt-5.4")
    client.usage.update(input_tokens=7, total_tokens=7)
    workflow.client = client
    counter = {**empty_usage(), "output_tokens": 3, "total_tokens": 3}

    assert total_usage(workflow, [a, {"b": b}], counter, counter) == {
        "input_tokens": 157,
        "output_tokens": 18,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 175,
    }


def test_total_usage_counts_a_member_that_keeps_a_usage_copy_once(
    monkeypatch,
):
    """A member storing a copy of its usage in its last run counts once."""
    step = {"id": "s1", "agent": "W", "prompt": "{task}"}
    plan = {
        "name": "p",
        "description": "d",
        "inline_agents": [{"name": "W", "system_prompt": "work"}],
        "phases": [{"title": "P1", "steps": [step]}],
        "return_steps": ["s1"],
    }
    answers = {
        "ProposeThoughts": {
            "thoughts": [{"content": "x", "is_final": True}]
        },
        "EvaluateThought": {"reasoning": "ok", "score": 1.0},
        "FinalAnswer": {"answer": "42"},
    }

    def run(self, task=None, *args, **kwargs):
        self._add_usage({"total_tokens": 15})
        if self.tools_list_dictionary:
            name = self.tools_list_dictionary[0]["function"]["name"]
            return {
                "function": {"name": name, "arguments": answers[name]}
            }
        if self.agent_name.endswith("-lead"):
            return json.dumps(plan)
        return "answer"

    monkeypatch.setattr(Agent, "run", run)

    for member in (
        PhasedFanOutWorkflow(name="Research"),
        TreeOfThoughts(max_depth=1, num_thoughts=1, breadth=1),
    ):
        writer = Agent(agent_name="Writer", print_on=False)
        workflow = SequentialWorkflow(agents=[writer, member])
        workflow.run("What is 6*7?")

        assert member.usage["total_tokens"] == 3 * 15
        assert total_usage(member) == member.usage
        assert workflow.usage["total_tokens"] == 4 * 15
