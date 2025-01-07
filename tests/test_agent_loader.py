import shutil
from pathlib import Path
from typing import Dict, Any

from swarms.structs.agent_loader import (
    AgentLoader,
    AgentValidator,
    AgentValidationError,
)


def setup_test_environment() -> Path:
    """Create a temporary test directory and return its path"""
    test_dir = Path("test_agents")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    return test_dir


def cleanup_test_environment(test_dir: Path) -> None:
    """Clean up the test directory"""
    if test_dir.exists():
        shutil.rmtree(test_dir)


def get_valid_agent_config() -> Dict[str, Any]:
    """Return a valid agent configuration for testing"""
    return {
        "agent_name": "Test-Agent",
        "system_prompt": "You are a test agent",
        "model_name": "gpt-4",
        "max_loops": 1,
        "autosave": True,
        "dashboard": False,
        "verbose": True,
        "dynamic_temperature": True,
        "saved_state_path": "test_agent.json",
        "user_name": "test_user",
        "retry_attempts": 3,
        "context_length": 200000,
        "return_step_meta": False,
        "output_type": "string",
        "streaming": False,
    }


def test_agent_validator():
    """Test AgentValidator functionality"""
    print("\nTesting AgentValidator...")

    # Test valid configuration
    config = get_valid_agent_config()
    try:
        AgentValidator.validate_config(config)
        print("✓ Valid configuration accepted")
    except AgentValidationError:
        print("✗ Valid configuration rejected")
        return False

    # Test invalid model name
    invalid_config = get_valid_agent_config()
    invalid_config["model_name"] = "invalid-model"
    try:
        AgentValidator.validate_config(invalid_config)
        print("✗ Invalid model name accepted")
        return False
    except AgentValidationError:
        print("✓ Invalid model name rejected")

    return True


def test_agent_loader_initialization():
    """Test AgentLoader initialization"""
    print("\nTesting AgentLoader initialization...")

    test_dir = setup_test_environment()

    try:
        # Test with string path
        AgentLoader("test_agents/agents")
        print("✓ Initialized with string path")

        # Test with Path object
        AgentLoader(Path("test_agents/agents"))
        print("✓ Initialized with Path object")

        # Verify directory creation
        if Path("test_agents/agents").parent.exists():
            print("✓ Parent directory created")
        else:
            print("✗ Parent directory not created")
            return False

    finally:
        cleanup_test_environment(test_dir)

    return True


def test_csv_creation_and_loading():
    """Test CSV creation and loading functionality"""
    print("\nTesting CSV creation and loading...")

    test_dir = setup_test_environment()

    try:
        loader = AgentLoader("test_agents/test_agents")
        config = get_valid_agent_config()

        # Test CSV creation
        try:
            loader.create_agents([config], file_type="csv")
            print("✓ CSV file created")
        except Exception as e:
            print(f"✗ CSV creation failed: {str(e)}")
            return False

        # Verify file exists
        if not Path("test_agents/test_agents.csv").exists():
            print("✗ CSV file not found")
            return False
        print("✓ CSV file exists")

        # Test CSV loading
        try:
            agents = loader.load_agents(file_type="csv")
            if len(agents) == 1:
                print("✓ Agents loaded from CSV")
            else:
                print("✗ Incorrect number of agents loaded")
                return False
        except Exception as e:
            print(f"✗ CSV loading failed: {str(e)}")
            return False

    finally:
        cleanup_test_environment(test_dir)

    return True


def test_json_creation_and_loading():
    """Test JSON creation and loading functionality"""
    print("\nTesting JSON creation and loading...")

    test_dir = setup_test_environment()

    try:
        loader = AgentLoader("test_agents/test_agents")
        config = get_valid_agent_config()

        # Test JSON creation
        try:
            loader.create_agents([config], file_type="json")
            print("✓ JSON file created")
        except Exception as e:
            print(f"✗ JSON creation failed: {str(e)}")
            return False

        # Verify file exists
        if not Path("test_agents/test_agents.json").exists():
            print("✗ JSON file not found")
            return False
        print("✓ JSON file exists")

        # Test JSON loading
        try:
            agents = loader.load_agents(file_type="json")
            if len(agents) == 1:
                print("✓ Agents loaded from JSON")
            else:
                print("✗ Incorrect number of agents loaded")
                return False
        except Exception as e:
            print(f"✗ JSON loading failed: {str(e)}")
            return False

    finally:
        cleanup_test_environment(test_dir)

    return True


def test_invalid_file_type():
    """Test handling of invalid file types"""
    print("\nTesting invalid file type handling...")

    test_dir = setup_test_environment()

    try:
        loader = AgentLoader("test_agents/test_agents")
        config = get_valid_agent_config()

        # Test invalid file type for creation
        try:
            loader.create_agents([config], file_type="yaml")
            print("✗ Invalid file type accepted")
            return False
        except ValueError:
            print("✓ Invalid file type rejected")

        # Test invalid file type for loading
        try:
            loader.load_agents(file_type="yaml")
            print("✗ Invalid file type accepted for loading")
            return False
        except ValueError:
            print("✓ Invalid file type rejected for loading")

    finally:
        cleanup_test_environment(test_dir)

    return True


def test_multiple_agents():
    """Test handling of multiple agents"""
    print("\nTesting multiple agents handling...")

    test_dir = setup_test_environment()

    try:
        loader = AgentLoader("test_agents/test_agents")

        # Create multiple agent configs
        configs = [
            get_valid_agent_config(),
            {
                **get_valid_agent_config(),
                "agent_name": "Test-Agent-2",
            },
            {
                **get_valid_agent_config(),
                "agent_name": "Test-Agent-3",
            },
        ]

        # Test CSV creation and loading
        try:
            loader.create_agents(configs, file_type="csv")
            agents = loader.load_agents(file_type="csv")
            if len(agents) == 3:
                print("✓ Multiple agents handled in CSV")
            else:
                print(
                    f"✗ Incorrect number of agents in CSV: {len(agents)}"
                )
                return False
        except Exception as e:
            print(f"✗ Multiple agents CSV test failed: {str(e)}")
            return False

        # Test JSON creation and loading
        try:
            loader.create_agents(configs, file_type="json")
            agents = loader.load_agents(file_type="json")
            if len(agents) == 3:
                print("✓ Multiple agents handled in JSON")
            else:
                print(
                    f"✗ Incorrect number of agents in JSON: {len(agents)}"
                )
                return False
        except Exception as e:
            print(f"✗ Multiple agents JSON test failed: {str(e)}")
            return False

    finally:
        cleanup_test_environment(test_dir)

    return True


def run_all_tests():
    """Run all test functions and report results"""
    tests = [
        test_agent_validator,
        test_agent_loader_initialization,
        test_csv_creation_and_loading,
        test_json_creation_and_loading,
        test_invalid_file_type,
        test_multiple_agents,
    ]

    total_tests = len(tests)
    passed_tests = 0

    print("Starting AgentLoader tests...\n")

    for test in tests:
        if test():
            passed_tests += 1

    print(
        f"\nTest Results: {passed_tests}/{total_tests} tests passed"
    )

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
