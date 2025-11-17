import os
import json
import csv

import pytest

from swarms.structs.agent import Agent
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory for test isolation."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return str(workspace)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file with agent configurations."""
    csv_path = tmp_path / "test_agents.csv"
    csv_content = [
        [
            "agent_name",
            "description",
            "system_prompt",
            "task",
            "model_name",
        ],
        [
            "agent_1",
            "First test agent",
            "You are a helpful assistant. Respond with exactly 'Task completed.'",
            "Say hello",
            "gpt-4o-mini",
        ],
        [
            "agent_2",
            "Second test agent",
            "You are a code reviewer. Respond with exactly 'Review done.'",
            "Review this: print('hello')",
            "gpt-4o-mini",
        ],
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_content)

    return str(csv_path)


def test_swarm_initialization_basic(temp_workspace):
    """Test basic swarm initialization with required parameters."""
    agent = Agent(
        agent_name="test_agent_1",
        system_prompt="You are a helpful assistant",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        name="Test Swarm",
        description="Test swarm description",
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    assert swarm.name == "Test Swarm"
    assert swarm.description == "Test swarm description"
    assert len(swarm.agents) == 1
    assert swarm.max_loops == 1
    assert swarm.autosave is True
    assert swarm.tasks_completed == 0
    assert swarm.outputs == []


def test_swarm_initialization_multiple_agents(temp_workspace):
    """Test swarm initialization with multiple agents."""
    agents = [
        Agent(
            agent_name="agent_1",
            system_prompt="You are agent 1",
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
        Agent(
            agent_name="agent_2",
            system_prompt="You are agent 2",
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
    ]

    swarm = SpreadSheetSwarm(
        name="Multi Agent Swarm",
        agents=agents,
        workspace_dir=temp_workspace,
    )

    assert len(swarm.agents) == 2
    assert swarm.agents[0].agent_name == "agent_1"
    assert swarm.agents[1].agent_name == "agent_2"


def test_swarm_initialization_custom_max_loops(temp_workspace):
    """Test initialization with custom max_loops setting."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        name="Custom Loop Swarm",
        agents=[agent],
        max_loops=3,
        workspace_dir=temp_workspace,
    )

    assert swarm.max_loops == 3


def test_swarm_initialization_autosave_disabled(temp_workspace):
    """Test initialization with autosave disabled."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        autosave=False,
        workspace_dir=temp_workspace,
    )

    assert swarm.autosave is False


def test_swarm_save_file_path_generation(temp_workspace):
    """Test that save file path is correctly generated with workspace_dir."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    assert swarm.save_file_path is not None
    assert "spreadsheet_swarm_run_id_" in swarm.save_file_path
    assert swarm.save_file_path.startswith(temp_workspace)
    assert swarm.save_file_path.endswith(".csv")


def test_swarm_initialization_no_agents_raises_error():
    """Test that initialization without agents raises ValueError."""
    with pytest.raises(ValueError, match="No agents are provided"):
        SpreadSheetSwarm(agents=None)


def test_swarm_initialization_no_max_loops_raises_error():
    """Test that initialization without max_loops raises ValueError."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    with pytest.raises(ValueError, match="No max loops are provided"):
        SpreadSheetSwarm(agents=[agent], max_loops=None)


def test_track_output_single(temp_workspace):
    """Test tracking a single task output."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    swarm._track_output("test_agent", "Test task", "Test result")

    assert swarm.tasks_completed == 1
    assert len(swarm.outputs) == 1
    assert swarm.outputs[0]["agent_name"] == "test_agent"
    assert swarm.outputs[0]["task"] == "Test task"
    assert swarm.outputs[0]["result"] == "Test result"
    assert "timestamp" in swarm.outputs[0]


def test_track_output_multiple(temp_workspace):
    """Test tracking multiple task outputs."""
    agents = [
        Agent(
            agent_name=f"agent_{i}",
            system_prompt="Test prompt",
            model_name="gpt-4o-mini",
            max_loops=1,
        )
        for i in range(1, 4)
    ]

    swarm = SpreadSheetSwarm(
        agents=agents,
        workspace_dir=temp_workspace,
    )

    for i in range(1, 4):
        swarm._track_output(f"agent_{i}", f"Task {i}", f"Result {i}")

    assert swarm.tasks_completed == 3
    assert len(swarm.outputs) == 3

    for i, output in enumerate(swarm.outputs, 1):
        assert output["agent_name"] == f"agent_{i}"
        assert output["task"] == f"Task {i}"
        assert output["result"] == f"Result {i}"


def test_track_output_increments_counter(temp_workspace):
    """Test that tasks_completed counter increments correctly."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    initial_count = swarm.tasks_completed
    swarm._track_output("test_agent", "Task 1", "Result 1")
    assert swarm.tasks_completed == initial_count + 1

    swarm._track_output("test_agent", "Task 2", "Result 2")
    assert swarm.tasks_completed == initial_count + 2


def test_load_from_csv_basic(sample_csv_file, temp_workspace):
    """Test loading agents from a CSV file."""
    # Initialize with empty agents list, will be populated from CSV
    agent = Agent(
        agent_name="placeholder",
        system_prompt="placeholder",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        load_path=sample_csv_file,
        workspace_dir=temp_workspace,
    )

    swarm._load_from_csv()

    # Should have loaded 2 agents from CSV plus the initial placeholder
    assert len(swarm.agents) == 3
    assert len(swarm.agent_tasks) == 2
    assert "agent_1" in swarm.agent_tasks
    assert "agent_2" in swarm.agent_tasks
    assert swarm.agent_tasks["agent_1"] == "Say hello"
    assert (
        swarm.agent_tasks["agent_2"] == "Review this: print('hello')"
    )


def test_load_from_csv_creates_agents(
    sample_csv_file, temp_workspace
):
    """Test that CSV loading creates proper Agent objects."""
    agent = Agent(
        agent_name="placeholder",
        system_prompt="placeholder",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        load_path=sample_csv_file,
        workspace_dir=temp_workspace,
    )

    swarm._load_from_csv()

    # Verify the loaded agents are proper Agent instances
    for agent in swarm.agents[1:]:  # Skip placeholder
        assert isinstance(agent, Agent)
        assert hasattr(agent, "agent_name")
        assert hasattr(agent, "system_prompt")


def test_load_from_nonexistent_csv(temp_workspace):
    """Test loading from non-existent CSV file."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        load_path="nonexistent_file.csv",
        workspace_dir=temp_workspace,
    )

    # Error is caught and logged, not raised
    swarm._load_from_csv()
    # Should still have the original agent
    assert len(swarm.agents) == 1


def test_save_to_csv_creates_file(temp_workspace):
    """Test that saving to CSV creates the output file."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    swarm._track_output("test_agent", "Test task", "Test result")
    swarm._save_to_csv()

    assert os.path.exists(swarm.save_file_path)


def test_save_to_csv_headers(temp_workspace):
    """Test that CSV file includes proper headers."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    swarm._track_output("test_agent", "Test task", "Test result")
    swarm._save_to_csv()

    with open(swarm.save_file_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers == [
            "Run ID",
            "Agent Name",
            "Task",
            "Result",
            "Timestamp",
        ]


def test_save_to_csv_data(temp_workspace):
    """Test that CSV file includes the tracked output data."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    swarm._track_output("test_agent", "Test task", "Test result")
    swarm._save_to_csv()

    with open(swarm.save_file_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["Agent Name"] == "test_agent"
        assert rows[0]["Task"] == "Test task"
        assert rows[0]["Result"] == "Test result"


def test_save_to_csv_appends(temp_workspace):
    """Test that multiple saves append to the same CSV file.

    Note: _save_to_csv() saves ALL outputs in swarm.outputs each time,
    so calling it twice will result in duplicates. This tests the actual behavior.
    """
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    swarm._track_output("test_agent", "Task 1", "Result 1")
    swarm._save_to_csv()

    swarm._track_output("test_agent", "Task 2", "Result 2")
    swarm._save_to_csv()

    # After first save: Task 1 (1 row)
    # After second save: Task 1 + Task 2 (2 more rows)
    # Total: 3 rows (Task 1 appears twice, Task 2 appears once)
    with open(swarm.save_file_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3

        # Verify the data
        assert rows[0]["Task"] == "Task 1"
        assert rows[1]["Task"] == "Task 1"
        assert rows[2]["Task"] == "Task 2"


def test_export_to_json_structure(temp_workspace):
    """Test that JSON export contains expected structure."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        name="JSON Test Swarm",
        description="Testing JSON export",
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    swarm._track_output("test_agent", "Test task", "Test result")
    json_output = swarm.export_to_json()
    data = json.loads(json_output)

    assert "run_id" in data
    assert "name" in data
    assert "description" in data
    assert "tasks_completed" in data
    assert "number_of_agents" in data
    assert "outputs" in data

    assert data["name"] == "JSON Test Swarm"
    assert data["description"] == "Testing JSON export"
    assert data["tasks_completed"] == 1
    assert data["number_of_agents"] == 1


def test_export_to_json_outputs(temp_workspace):
    """Test that JSON export includes all tracked outputs."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    swarm._track_output("test_agent", "Task 1", "Result 1")
    swarm._track_output("test_agent", "Task 2", "Result 2")

    json_output = swarm.export_to_json()
    data = json.loads(json_output)

    assert len(data["outputs"]) == 2
    assert data["outputs"][0]["agent_name"] == "test_agent"
    assert data["outputs"][0]["task"] == "Task 1"
    assert data["outputs"][1]["task"] == "Task 2"


def test_export_to_json_valid_format(temp_workspace):
    """Test that JSON export is valid JSON."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    json_output = swarm.export_to_json()
    data = json.loads(json_output)
    assert isinstance(data, dict)


def test_export_empty_swarm_to_json(temp_workspace):
    """Test JSON export with no completed tasks."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        workspace_dir=temp_workspace,
    )

    json_output = swarm.export_to_json()
    data = json.loads(json_output)

    assert data["tasks_completed"] == 0
    assert data["outputs"] == []
    assert data["number_of_agents"] == 1


def test_reliability_check_passes(temp_workspace):
    """Test that reliability check passes with valid configuration."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        max_loops=1,
        workspace_dir=temp_workspace,
    )

    assert swarm is not None


def test_reliability_check_verbose(temp_workspace):
    """Test verbose mode during initialization."""
    agent = Agent(
        agent_name="test_agent",
        system_prompt="Test prompt",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    swarm = SpreadSheetSwarm(
        agents=[agent],
        verbose=True,
        workspace_dir=temp_workspace,
    )

    assert swarm.verbose is True
