from unittest.mock import patch
from swarms.swarms.multi_agent_collab import (
    MultiAgentCollaboration,
    Worker,
    select_next_speaker,
)


def test_multiagentcollaboration_initialization():
    multiagentcollaboration = MultiAgentCollaboration(
        agents=[Worker] * 5, selection_function=select_next_speaker
    )
    assert isinstance(multiagentcollaboration, MultiAgentCollaboration)
    assert len(multiagentcollaboration.agents) == 5
    assert multiagentcollaboration._step == 0


@patch("swarms.workers.Worker.reset")
def test_multiagentcollaboration_reset(mock_reset):
    multiagentcollaboration = MultiAgentCollaboration(
        agents=[Worker] * 5, selection_function=select_next_speaker
    )
    multiagentcollaboration.reset()
    assert mock_reset.call_count == 5


@patch("swarms.workers.Worker.run")
def test_multiagentcollaboration_inject(mock_run):
    multiagentcollaboration = MultiAgentCollaboration(
        agents=[Worker] * 5, selection_function=select_next_speaker
    )
    multiagentcollaboration.inject("Agent 1", "Hello, world!")
    assert multiagentcollaboration._step == 1
    assert mock_run.call_count == 5


@patch("swarms.workers.Worker.send")
@patch("swarms.workers.Worker.receive")
def test_multiagentcollaboration_step(mock_receive, mock_send):
    multiagentcollaboration = MultiAgentCollaboration(
        agents=[Worker] * 5, selection_function=select_next_speaker
    )
    multiagentcollaboration.step()
    assert multiagentcollaboration._step == 1
    assert mock_send.call_count == 5
    assert mock_receive.call_count == 25


@patch("swarms.workers.Worker.bid")
def test_multiagentcollaboration_ask_for_bid(mock_bid):
    multiagentcollaboration = MultiAgentCollaboration(
        agents=[Worker] * 5, selection_function=select_next_speaker
    )
    result = multiagentcollaboration.ask_for_bid(Worker)
    assert isinstance(result, int)


@patch("swarms.workers.Worker.bid")
def test_multiagentcollaboration_select_next_speaker(mock_bid):
    multiagentcollaboration = MultiAgentCollaboration(
        agents=[Worker] * 5, selection_function=select_next_speaker
    )
    result = multiagentcollaboration.select_next_speaker(1, [Worker] * 5)
    assert isinstance(result, int)


@patch("swarms.workers.Worker.send")
@patch("swarms.workers.Worker.receive")
def test_multiagentcollaboration_run(mock_receive, mock_send):
    multiagentcollaboration = MultiAgentCollaboration(
        agents=[Worker] * 5, selection_function=select_next_speaker
    )
    multiagentcollaboration.run(max_iters=5)
    assert multiagentcollaboration._step == 6
    assert mock_send.call_count == 30
    assert mock_receive.call_count == 150
