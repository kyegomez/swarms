import requests
from loguru import logger
import time
from typing import Dict, Optional, Tuple
from uuid import UUID

BASE_URL = "http://0.0.0.0:8000/v1"


def check_api_server() -> bool:
    """Check if the API server is running and accessible."""
    try:
        response = requests.get(f"{BASE_URL}/docs")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        logger.error("API server is not running at {BASE_URL}")
        logger.error("Please start the API server first with:")
        logger.error("    python main.py")
        return False
    except Exception as e:
        logger.error(f"Error checking API server: {str(e)}")
        return False


class TestSession:
    """Manages test session state and authentication."""

    def __init__(self):
        self.user_id: Optional[UUID] = None
        self.api_key: Optional[str] = None
        self.test_agents: list[UUID] = []

    @property
    def headers(self) -> Dict[str, str]:
        """Get headers with authentication."""
        return {"api-key": self.api_key} if self.api_key else {}


def create_test_user(session: TestSession) -> Tuple[bool, str]:
    """Create a test user and store credentials in session."""
    logger.info("Creating test user")

    try:
        response = requests.post(
            f"{BASE_URL}/users",
            json={"username": f"test_user_{int(time.time())}"},
        )

        if response.status_code == 200:
            data = response.json()
            session.user_id = data["user_id"]
            session.api_key = data["api_key"]
            logger.success(f"Created user with ID: {session.user_id}")
            return True, "Success"
        else:
            logger.error(f"Failed to create user: {response.text}")
            return False, response.text
    except Exception as e:
        logger.exception("Exception during user creation")
        return False, str(e)


def create_additional_api_key(
    session: TestSession,
) -> Tuple[bool, str]:
    """Test creating an additional API key."""
    logger.info("Creating additional API key")

    try:
        response = requests.post(
            f"{BASE_URL}/users/{session.user_id}/api-keys",
            headers=session.headers,
            json={"name": "Test Key"},
        )

        if response.status_code == 200:
            logger.success("Created additional API key")
            return True, response.json()["key"]
        else:
            logger.error(f"Failed to create API key: {response.text}")
            return False, response.text
    except Exception as e:
        logger.exception("Exception during API key creation")
        return False, str(e)


def test_create_agent(
    session: TestSession,
) -> Tuple[bool, Optional[UUID]]:
    """Test creating a new agent."""
    logger.info("Testing agent creation")

    payload = {
        "agent_name": f"Test Agent {int(time.time())}",
        "system_prompt": "You are a helpful assistant",
        "model_name": "gpt-4",
        "description": "Test agent",
        "tags": ["test", "automated"],
    }

    try:
        response = requests.post(
            f"{BASE_URL}/agent", headers=session.headers, json=payload
        )

        if response.status_code == 200:
            agent_id = response.json()["agent_id"]
            session.test_agents.append(agent_id)
            logger.success(f"Created agent with ID: {agent_id}")
            return True, agent_id
        else:
            logger.error(f"Failed to create agent: {response.text}")
            return False, None
    except Exception:
        logger.exception("Exception during agent creation")
        return False, None


def test_list_user_agents(session: TestSession) -> bool:
    """Test listing user's agents."""
    logger.info("Testing user agent listing")

    try:
        response = requests.get(
            f"{BASE_URL}/users/me/agents", headers=session.headers
        )

        if response.status_code == 200:
            agents = response.json()
            logger.success(f"Found {len(agents)} user agents")
            return True
        else:
            logger.error(
                f"Failed to list user agents: {response.text}"
            )
            return False
    except Exception:
        logger.exception("Exception during agent listing")
        return False


def test_agent_operations(
    session: TestSession, agent_id: UUID
) -> bool:
    """Test various operations on an agent."""
    logger.info(f"Testing operations for agent {agent_id}")

    # Test update
    try:
        update_response = requests.patch(
            f"{BASE_URL}/agent/{agent_id}",
            headers=session.headers,
            json={
                "description": "Updated description",
                "tags": ["test", "updated"],
            },
        )
        if update_response.status_code != 200:
            logger.error(
                f"Failed to update agent: {update_response.text}"
            )
            return False

        # Test metrics
        metrics_response = requests.get(
            f"{BASE_URL}/agent/{agent_id}/metrics",
            headers=session.headers,
        )
        if metrics_response.status_code != 200:
            logger.error(
                f"Failed to get agent metrics: {metrics_response.text}"
            )
            return False

        logger.success("Successfully performed agent operations")
        return True
    except Exception:
        logger.exception("Exception during agent operations")
        return False


def test_completion(session: TestSession, agent_id: UUID) -> bool:
    """Test running a completion."""
    logger.info("Testing completion")

    payload = {
        "prompt": "What is the weather like today?",
        "agent_id": agent_id,
        "max_tokens": 100,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/agent/completions",
            headers=session.headers,
            json=payload,
        )

        if response.status_code == 200:
            completion_data = response.json()
            print(completion_data)
            logger.success(
                f"Got completion, used {completion_data['token_usage']['total_tokens']} tokens"
            )
            return True
        else:
            logger.error(f"Failed to get completion: {response.text}")
            return False
    except Exception:
        logger.exception("Exception during completion")
        return False


def cleanup_test_resources(session: TestSession):
    """Clean up all test resources."""
    logger.info("Cleaning up test resources")

    # Delete test agents
    for agent_id in session.test_agents:
        try:
            response = requests.delete(
                f"{BASE_URL}/agent/{agent_id}",
                headers=session.headers,
            )
            if response.status_code == 200:
                logger.debug(f"Deleted agent {agent_id}")
            else:
                logger.warning(
                    f"Failed to delete agent {agent_id}: {response.text}"
                )
        except Exception:
            logger.exception(f"Exception deleting agent {agent_id}")

    # Revoke API keys
    if session.user_id:
        try:
            response = requests.get(
                f"{BASE_URL}/users/{session.user_id}/api-keys",
                headers=session.headers,
            )
            if response.status_code == 200:
                for key in response.json():
                    try:
                        revoke_response = requests.delete(
                            f"{BASE_URL}/users/{session.user_id}/api-keys/{key['key']}",
                            headers=session.headers,
                        )
                        if revoke_response.status_code == 200:
                            logger.debug(
                                f"Revoked API key {key['name']}"
                            )
                        else:
                            logger.warning(
                                f"Failed to revoke API key {key['name']}"
                            )
                    except Exception:
                        logger.exception(
                            f"Exception revoking API key {key['name']}"
                        )
        except Exception:
            logger.exception("Exception getting API keys for cleanup")


def run_test_workflow():
    """Run complete test workflow."""
    logger.info("Starting API tests")

    # Check if API server is running first
    if not check_api_server():
        return False

    session = TestSession()

    try:
        # Create user
        user_success, message = create_test_user(session)
        if not user_success:
            logger.error(f"User creation failed: {message}")
            return False

        # Create additional API key
        key_success, key = create_additional_api_key(session)
        if not key_success:
            logger.error(f"API key creation failed: {key}")
            return False

        # Create agent
        agent_success, agent_id = test_create_agent(session)
        if not agent_success or not agent_id:
            logger.error("Agent creation failed")
            return False

        # Test user agent listing
        if not test_list_user_agents(session):
            logger.error("Agent listing failed")
            return False

        # Test agent operations
        if not test_agent_operations(session, agent_id):
            logger.error("Agent operations failed")
            return False

        # Test completion
        if not test_completion(session, agent_id):
            logger.error("Completion test failed")
            return False

        logger.success("All tests completed successfully")
        return True

    except Exception:
        logger.exception("Exception during test workflow")
        return False
    finally:
        cleanup_test_resources(session)


if __name__ == "__main__":
    success = run_test_workflow()
    print(success)
