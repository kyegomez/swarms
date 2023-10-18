from unittest.mock import patch
from swarms.swarms.multi_agent_debate import MultiAgentDebate, Worker, select_speaker


def test_multiagentdebate_initialization():
    multiagentdebate = MultiAgentDebate(
        agents=[Worker] * 5, selection_func=select_speaker
    )
    assert isinstance(multiagentdebate, MultiAgentDebate)
    assert len(multiagentdebate.agents) == 5
    assert multiagentdebate.selection_func == select_speaker


@patch("swarms.workers.Worker.reset")
def test_multiagentdebate_reset_agents(mock_reset):
    multiagentdebate = MultiAgentDebate(
        agents=[Worker] * 5, selection_func=select_speaker
    )
    multiagentdebate.reset_agents()
    assert mock_reset.call_count == 5


def test_multiagentdebate_inject_agent():
    multiagentdebate = MultiAgentDebate(
        agents=[Worker] * 5, selection_func=select_speaker
    )
    multiagentdebate.inject_agent(Worker)
    assert len(multiagentdebate.agents) == 6


@patch("swarms.workers.Worker.run")
def test_multiagentdebate_run(mock_run):
    multiagentdebate = MultiAgentDebate(
        agents=[Worker] * 5, selection_func=select_speaker
    )
    results = multiagentdebate.run("Write a short story.")
    assert len(results) == 5
    assert mock_run.call_count == 5


def test_multiagentdebate_update_task():
    multiagentdebate = MultiAgentDebate(
        agents=[Worker] * 5, selection_func=select_speaker
    )
    multiagentdebate.update_task("Write a short story.")
    assert multiagentdebate.task == "Write a short story."


def test_multiagentdebate_format_results():
    multiagentdebate = MultiAgentDebate(
        agents=[Worker] * 5, selection_func=select_speaker
    )
    results = [
        {"agent": "Agent 1", "response": "Hello, world!"},
        {"agent": "Agent 2", "response": "Goodbye, world!"},
    ]
    formatted_results = multiagentdebate.format_results(results)
    assert (
        formatted_results
        == "Agent Agent 1 responded: Hello, world!\nAgent Agent 2 responded: Goodbye, world!"
    )
