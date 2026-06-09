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

    with patch("swarms.structs.mixture_of_agents.run_agents_concurrently") as mock_run:
        mock_run.return_value = {w.agent_name: w.agent_name + "-output" for w in workers}
        moa = MixtureOfAgents(agents=workers, aggregator_agent=aggregator, layers=layers)
        moa._run_concurrent_mock = mock_run
    return moa


def test_layer0_workers_receive_only_task():
    """Workers on the first layer should get the raw task, not the full transcript."""
    calls = []

    def fake_run(agents, task, img, return_agent_output_dict):
        calls.append(task)
        return {a.agent_name: "out" for a in agents}

    moa = _make_moa(layers=1)
    moa.aggregator_agent.run = MagicMock(return_value="agg")

    with patch("swarms.structs.mixture_of_agents.run_agents_concurrently", side_effect=fake_run):
        moa._run(task="my task")

    assert calls[0] == "my task"


def test_layer1_workers_receive_task_and_previous_synthesis():
    """Workers on layer 1+ should get task + previous-layer output, not full history."""
    calls = []

    def fake_run(agents, task, img, return_agent_output_dict):
        calls.append(task)
        return {a.agent_name: f"layer{len(calls)-1}-out" for a in agents}

    moa = _make_moa(layers=2)
    moa.aggregator_agent.run = MagicMock(return_value="agg")

    with patch("swarms.structs.mixture_of_agents.run_agents_concurrently", side_effect=fake_run):
        moa._run(task="my task")

    assert calls[0] == "my task"
    assert "my task" in calls[1]
    assert "Previous layer synthesis" in calls[1]
    # Must NOT contain role prefixes from earlier turns that would indicate full transcript
    assert "User:" not in calls[1]


def test_aggregator_still_receives_full_conversation():
    """The aggregator must see the full conversation, not just the last layer."""
    def fake_run(agents, task, img, return_agent_output_dict):
        return {a.agent_name: "worker-out" for a in agents}

    moa = _make_moa(layers=2)
    agg_calls = []
    moa.aggregator_agent.run = MagicMock(side_effect=lambda task: agg_calls.append(task) or "agg")

    with patch("swarms.structs.mixture_of_agents.run_agents_concurrently", side_effect=fake_run):
        moa._run(task="my task")

    # The aggregator input should contain more than just the bare task
    assert len(agg_calls[0]) > len("my task")
