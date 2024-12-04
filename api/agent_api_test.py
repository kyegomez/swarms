import requests
from loguru import logger
import time

# Configure loguru
logger.add(
    "api_tests_{time}.log",
    rotation="100 MB",
    level="DEBUG",
    format="{time} {level} {message}",
)

BASE_URL = "http://localhost:8000/v1"


def test_create_agent():
    """Test creating a new agent."""
    logger.info("Testing agent creation")

    payload = {
        "agent_name": "Test Agent",
        "system_prompt": "You are a helpful assistant",
        "model_name": "gpt-4",
        "description": "Test agent",
        "tags": ["test"],
    }

    response = requests.post(f"{BASE_URL}/agent", json=payload)
    logger.debug(f"Create response: {response.json()}")

    if response.status_code == 200:
        logger.success("Successfully created agent")
        return response.json()["agent_id"]
    else:
        logger.error(f"Failed to create agent: {response.text}")
        return None


def test_list_agents():
    """Test listing all agents."""
    logger.info("Testing agent listing")

    response = requests.get(f"{BASE_URL}/agents")
    logger.debug(f"List response: {response.json()}")

    if response.status_code == 200:
        logger.success(f"Found {len(response.json())} agents")
    else:
        logger.error(f"Failed to list agents: {response.text}")


def test_completion(agent_id):
    """Test running a completion."""
    logger.info("Testing completion")

    payload = {
        "prompt": "What is the weather like today?",
        "agent_id": agent_id,
    }

    response = requests.post(
        f"{BASE_URL}/agent/completions", json=payload
    )
    logger.debug(f"Completion response: {response.json()}")

    if response.status_code == 200:
        logger.success("Successfully got completion")
    else:
        logger.error(f"Failed to get completion: {response.text}")


def test_delete_agent(agent_id):
    """Test deleting an agent."""
    logger.info("Testing agent deletion")

    response = requests.delete(f"{BASE_URL}/agent/{agent_id}")
    logger.debug(f"Delete response: {response.json()}")

    if response.status_code == 200:
        logger.success("Successfully deleted agent")
    else:
        logger.error(f"Failed to delete agent: {response.text}")


def run_tests():
    """Run all tests in sequence."""
    logger.info("Starting API tests")

    # Create agent and get ID
    agent_id = test_create_agent()
    if not agent_id:
        logger.error("Cannot continue tests without agent ID")
        return

    # Wait a bit for agent to be ready
    time.sleep(1)

    # Run other tests
    test_list_agents()
    test_completion(agent_id)
    test_delete_agent(agent_id)

    logger.info("Tests completed")


if __name__ == "__main__":
    run_tests()
