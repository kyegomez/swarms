import os
import asyncio
from loguru import logger
from swarms.structs.agent import Agent
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm


def create_test_csv() -> str:
    """Create a test CSV file with agent configurations."""
    print("\nStarting creation of test CSV file")
    try:
        csv_content = """agent_name,description,system_prompt,task
test_agent_1,Test Agent 1,System prompt 1,Task 1
test_agent_2,Test Agent 2,System prompt 2,Task 2"""

        file_path = "test_agents.csv"
        with open(file_path, "w") as f:
            f.write(csv_content)

        print(f"Created CSV with content:\n{csv_content}")
        print(f"CSV file created at: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to create test CSV: {str(e)}")
        raise


def create_test_agent(name: str) -> Agent:
    """Create a test agent with specified name."""
    print(f"\nCreating test agent: {name}")
    try:
        agent = Agent(
            agent_name=name,
            system_prompt=f"Test prompt for {name}",
            model_name="gpt-4o-mini",
            max_loops=1,
            autosave=True,
            verbose=True,
        )
        print(f"Created agent: {name}")
        return agent
    except Exception as e:
        logger.error(f"Failed to create agent {name}: {str(e)}")
        raise


def test_swarm_initialization() -> None:
    """Test basic swarm initialization."""
    print("\n[TEST] Starting swarm initialization test")
    try:
        print("Creating test agents...")
        agents = [
            create_test_agent("agent1"),
            create_test_agent("agent2"),
        ]

        print("Initializing swarm...")
        swarm = SpreadSheetSwarm(
            name="Test Swarm",
            description="Test Description",
            agents=agents,
            max_loops=2,
        )

        print("Verifying swarm configuration...")
        assert swarm.name == "Test Swarm"
        assert swarm.description == "Test Description"
        assert len(swarm.agents) == 2
        assert swarm.max_loops == 2

        print("✅ Swarm initialization test PASSED")
    except Exception as e:
        logger.error(f"❌ Swarm initialization test FAILED: {str(e)}")
        raise


async def test_load_from_csv() -> None:
    """Test loading agent configurations from CSV."""
    print("\n[TEST] Starting CSV loading test")
    try:
        csv_path = create_test_csv()
        print("Initializing swarm with CSV...")
        swarm = SpreadSheetSwarm(load_path=csv_path)

        print("Loading configurations...")
        await swarm._load_from_csv()

        print("Verifying loaded configurations...")
        assert len(swarm.agents) == 2
        assert len(swarm.agent_configs) == 2
        assert "test_agent_1" in swarm.agent_configs
        assert "test_agent_2" in swarm.agent_configs

        os.remove(csv_path)
        print(f"Cleaned up test file: {csv_path}")

        print("✅ CSV loading test PASSED")
    except Exception as e:
        logger.error(f"❌ CSV loading test FAILED: {str(e)}")
        raise


async def test_run_tasks() -> None:
    """Test running tasks with multiple agents."""
    print("\n[TEST] Starting task execution test")
    try:
        print("Setting up test swarm...")
        agents = [
            create_test_agent("agent1"),
            create_test_agent("agent2"),
        ]
        swarm = SpreadSheetSwarm(agents=agents, max_loops=1)

        test_task = "Test task for all agents"
        print(f"Running test task: {test_task}")
        await swarm._run_tasks(test_task)

        print("Verifying task execution...")
        assert swarm.metadata.tasks_completed == 2
        assert len(swarm.metadata.outputs) == 2

        print("✅ Task execution test PASSED")
    except Exception as e:
        logger.error(f"❌ Task execution test FAILED: {str(e)}")
        raise


def test_output_tracking() -> None:
    """Test tracking of task outputs."""
    print("\n[TEST] Starting output tracking test")
    try:
        print("Creating test swarm...")
        swarm = SpreadSheetSwarm(agents=[create_test_agent("agent1")])

        print("Tracking test output...")
        swarm._track_output("agent1", "Test task", "Test result")

        print("Verifying output tracking...")
        assert swarm.metadata.tasks_completed == 1
        assert len(swarm.metadata.outputs) == 1
        assert swarm.metadata.outputs[0].agent_name == "agent1"

        print("✅ Output tracking test PASSED")
    except Exception as e:
        logger.error(f"❌ Output tracking test FAILED: {str(e)}")
        raise


async def test_save_to_csv() -> None:
    """Test saving metadata to CSV."""
    print("\n[TEST] Starting CSV saving test")
    try:
        print("Setting up test data...")
        swarm = SpreadSheetSwarm(
            agents=[create_test_agent("agent1")],
            save_file_path="test_output.csv",
        )
        swarm._track_output("agent1", "Test task", "Test result")

        print("Saving to CSV...")
        await swarm._save_to_csv()

        print("Verifying file creation...")
        assert os.path.exists(swarm.save_file_path)

        os.remove(swarm.save_file_path)
        print("Cleaned up test file")

        print("✅ CSV saving test PASSED")
    except Exception as e:
        logger.error(f"❌ CSV saving test FAILED: {str(e)}")
        raise


def test_json_export() -> None:
    """Test JSON export functionality."""
    print("\n[TEST] Starting JSON export test")
    try:
        print("Creating test data...")
        swarm = SpreadSheetSwarm(agents=[create_test_agent("agent1")])
        swarm._track_output("agent1", "Test task", "Test result")

        print("Exporting to JSON...")
        json_output = swarm.export_to_json()

        print("Verifying JSON output...")
        assert isinstance(json_output, str)
        assert "run_id" in json_output
        assert "tasks_completed" in json_output

        print("✅ JSON export test PASSED")
    except Exception as e:
        logger.error(f"❌ JSON export test FAILED: {str(e)}")
        raise


async def run_all_tests() -> None:
    """Run all test functions."""
    print("\n" + "=" * 50)
    print("Starting SpreadsheetSwarm Test Suite")
    print("=" * 50 + "\n")

    try:
        # Run synchronous tests
        print("Running synchronous tests...")
        test_swarm_initialization()
        test_output_tracking()
        test_json_export()

        # Run asynchronous tests
        print("\nRunning asynchronous tests...")
        await test_load_from_csv()
        await test_run_tasks()
        await test_save_to_csv()

        print("\n🎉 All tests completed successfully!")
        print("=" * 50)
    except Exception as e:
        logger.error(f"\n❌ Test suite failed: {str(e)}")
        print("=" * 50)
        raise


if __name__ == "__main__":
    # Run all tests
    asyncio.run(run_all_tests())
