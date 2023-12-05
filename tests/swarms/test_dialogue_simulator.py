from unittest.mock import patch
from swarms.swarms.dialogue_simulator import DialogueSimulator, Worker


def test_dialoguesimulator_initialization():
    dialoguesimulator = DialogueSimulator(agents=[Worker] * 5)
    assert isinstance(dialoguesimulator, DialogueSimulator)
    assert len(dialoguesimulator.agents) == 5


@patch("swarms.workers.worker.Worker.run")
def test_dialoguesimulator_run(mock_run):
    dialoguesimulator = DialogueSimulator(agents=[Worker] * 5)
    dialoguesimulator.run(
        max_iters=5, name="Agent 1", message="Hello, world!"
    )
    assert mock_run.call_count == 30


@patch("swarms.workers.worker.Worker.run")
def test_dialoguesimulator_run_without_name_and_message(mock_run):
    dialoguesimulator = DialogueSimulator(agents=[Worker] * 5)
    dialoguesimulator.run(max_iters=5)
    assert mock_run.call_count == 25
