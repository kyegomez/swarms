"""Tests for MoA worker context fix (issue: workers were receiving full transcript)."""

from unittest.mock import MagicMock, patch

from swarms.structs.mixture_of_agents import MixtureOfAgents


def _make_moa(layers: int = 2) -> MixtureOfAgents:
    def _agent(name):
        a = MagicMock()
        a.agent_name = name
        a.run = MagicMock(return_value=f"{name}-output")
        return a

    workers = [_agent(f"Worker-{i}") for i in range(2)]
    aggregator = _agent("Aggregator")
    return MixtureOfAgents(
        agents=workers, aggregator_agent=aggregator, layers=layers
    )


def test_layer0_workers_receive_only_task():
    """Workers on the first layer should get the raw task, not the full transcript."""
    calls = []

    def fake_run(agents, task, img, return_agent_output_dict):
        calls.append(task)
        return {a.agent_name: "out" for a in agents}

    moa = _make_moa(layers=1)
    moa.aggregator_agent.run = MagicMock(return_value="agg")

    with patch(
        "swarms.structs.mixture_of_agents.run_agents_concurrently",
        side_effect=fake_run,
    ):
        moa._run(task="my task")

    assert calls[0] == "my task"


def test_layer1_workers_receive_task_and_previous_synthesis():
    """Workers on layer 1+ should get task + previous-layer output, not full history."""
    calls = []

    def fake_run(agents, task, img, return_agent_output_dict):
        calls.append(task)
        return {
            a.agent_name: f"layer{len(calls) - 1}-out" for a in agents
        }

    moa = _make_moa(layers=2)
    moa.aggregator_agent.run = MagicMock(return_value="agg")

    with patch(
        "swarms.structs.mixture_of_agents.run_agents_concurrently",
        side_effect=fake_run,
    ):
        moa._run(task="my task")

    assert calls[0] == "my task"
    assert "my task" in calls[1]
    assert "Previous layer synthesis" in calls[1]
    assert "User:" not in calls[1]


def test_aggregator_receives_worker_outputs_in_conversation():
    """The aggregator input must contain worker outputs, confirming it sees the full conversation."""

    def fake_run(agents, task, img, return_agent_output_dict):
        return {a.agent_name: "worker-out" for a in agents}

    moa = _make_moa(layers=1)
    agg_calls = []
    moa.aggregator_agent.run = MagicMock(
        side_effect=lambda task: agg_calls.append(task) or "agg"
    )

    with patch(
        "swarms.structs.mixture_of_agents.run_agents_concurrently",
        side_effect=fake_run,
    ):
        moa._run(task="my task")

    assert "worker-out" in agg_calls[0]
