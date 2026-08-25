"""Tests for swarms.structs.multi_agent_exec."""

from swarms.structs.multi_agent_exec import run_agents_concurrently


class _Stub:
    """Minimal stand-in for an Agent: a name and a fixed answer."""

    def __init__(self, agent_name: str, answer: str):
        self.agent_name = agent_name
        self._answer = answer

    def run(self, task=None, **kwargs):
        return self._answer


def test_output_dict_keeps_one_entry_per_agent_when_names_collide():
    """Agent.agent_name defaults to the same string for every agent.

    A swarm whose agents never set one used to collapse into a single dict
    entry, so MixtureOfAgents recorded one worker per layer and aggregated a
    single opinion instead of the mixture it was asked for.
    """
    agents = [
        _Stub("swarm-worker-01", f"answer-{i}") for i in range(3)
    ]

    outputs = run_agents_concurrently(
        agents=agents, task="t", return_agent_output_dict=True
    )

    assert len(outputs) == 3, outputs
    assert sorted(outputs.values()) == [
        "answer-0",
        "answer-1",
        "answer-2",
    ]


def test_output_dict_leaves_distinct_names_untouched():
    agents = [_Stub(f"worker-{i}", f"answer-{i}") for i in range(3)]

    outputs = run_agents_concurrently(
        agents=agents, task="t", return_agent_output_dict=True
    )

    assert outputs == {
        "worker-0": "answer-0",
        "worker-1": "answer-1",
        "worker-2": "answer-2",
    }
