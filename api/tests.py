import asyncio
import json
import os
import sys
from typing import Any, Dict

import aiohttp
from loguru import logger

# Configure loguru
LOG_PATH = "api_tests.log"
logger.add(
    LOG_PATH,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)

BASE_URL = (
    "https://api.swarms.ai/v1"  # Change this to match your server URL
)


async def log_request_details(
    method: str, url: str, headers: dict, data: Any = None
):
    """Log request details before sending."""
    logger.debug(f"\n{'='*50}")
    logger.debug(f"REQUEST: {method} {url}")
    logger.debug(f"HEADERS: {json.dumps(headers, indent=2)}")
    if data:
        logger.debug(f"PAYLOAD: {json.dumps(data, indent=2)}")


async def log_response_details(
    response: aiohttp.ClientResponse, data: Any = None
):
    """Log response details after receiving."""
    logger.debug(f"\nRESPONSE Status: {response.status}")
    logger.debug(
        f"RESPONSE Headers: {json.dumps(dict(response.headers), indent=2)}"
    )
    if data:
        logger.debug(f"RESPONSE Body: {json.dumps(data, indent=2)}")
    logger.debug(f"{'='*50}\n")


async def test_create_user(
    session: aiohttp.ClientSession,
) -> Dict[str, str]:
    """Test user creation endpoint."""
    url = f"{BASE_URL}/users"
    payload = {"username": "test_user"}

    logger.info("Testing user creation...")
    await log_request_details("POST", url, {}, payload)

    try:
        async with session.post(url, json=payload) as response:
            data = await response.json()
            await log_response_details(response, data)

            if response.status != 200:
                logger.error(
                    f"Failed to create user. Status: {response.status}, Response: {data}"
                )
                sys.exit(1)

            logger.success("✓ Created user successfully")
            return {
                "user_id": data["user_id"],
                "api_key": data["api_key"],
            }
    except Exception as e:
        logger.exception(f"Exception in user creation: {str(e)}")
        sys.exit(1)


async def test_create_agent(
    session: aiohttp.ClientSession, api_key: str
) -> str:
    """Test agent creation endpoint."""
    url = f"{BASE_URL}/agent"
    config = {
        "agent_name": "test_agent",
        "system_prompt": "You are a helpful test agent",
        "model_name": "gpt-4",
        "description": "Test agent for API validation",
        "max_loops": 1,
        "temperature": 0.5,
        "tags": ["test"],
        "streaming_on": False,
        "user_name": "test_user",  # Added required field
        "output_type": "string",  # Added required field
    }

    headers = {"api-key": api_key}
    logger.info("Testing agent creation...")
    await log_request_details("POST", url, headers, config)

    try:
        async with session.post(
            url, headers=headers, json=config
        ) as response:
            data = await response.json()
            await log_response_details(response, data)

            if response.status != 200:
                logger.error(
                    f"Failed to create agent. Status: {response.status}, Response: {data}"
                )
                return None

            logger.success("✓ Created agent successfully")
            return data["agent_id"]
    except Exception as e:
        logger.exception(f"Exception in agent creation: {str(e)}")
        return None


async def test_agent_update(
    session: aiohttp.ClientSession, agent_id: str, api_key: str
):
    """Test agent update endpoint."""
    url = f"{BASE_URL}/agent/{agent_id}"
    update_data = {
        "description": "Updated test agent",
        "system_prompt": "Updated system prompt",
        "temperature": 0.7,
        "tags": ["test", "updated"],
    }

    headers = {"api-key": api_key}
    logger.info(f"Testing agent update for agent {agent_id}...")
    await log_request_details("PATCH", url, headers, update_data)

    try:
        async with session.patch(
            url, headers=headers, json=update_data
        ) as response:
            data = await response.json()
            await log_response_details(response, data)

            if response.status != 200:
                logger.error(
                    f"Failed to update agent. Status: {response.status}, Response: {data}"
                )
                return False

            logger.success("✓ Updated agent successfully")
            return True
    except Exception as e:
        logger.exception(f"Exception in agent update: {str(e)}")
        return False


async def test_completion(
    session: aiohttp.ClientSession, agent_id: str, api_key: str
):
    """Test completion endpoint."""
    url = f"{BASE_URL}/agent/completions"
    completion_request = {
        "prompt": "Hello, how are you?",
        "agent_id": agent_id,
        "max_tokens": 100,
        "stream": False,
    }

    headers = {"api-key": api_key}
    logger.info(f"Testing completion for agent {agent_id}...")
    await log_request_details(
        "POST", url, headers, completion_request
    )

    try:
        async with session.post(
            url, headers=headers, json=completion_request
        ) as response:
            data = await response.json()
            await log_response_details(response, data)

            if response.status != 200:
                logger.error(
                    f"Failed to process completion. Status: {response.status}, Response: {data}"
                )
                return False

            logger.success("✓ Processed completion successfully")
            return True
    except Exception as e:
        logger.exception(
            f"Exception in completion processing: {str(e)}"
        )
        return False


async def test_get_metrics(
    session: aiohttp.ClientSession, agent_id: str, api_key: str
):
    """Test metrics endpoint."""
    url = f"{BASE_URL}/agent/{agent_id}/metrics"
    headers = {"api-key": api_key}

    logger.info(f"Testing metrics retrieval for agent {agent_id}...")
    await log_request_details("GET", url, headers)

    try:
        async with session.get(url, headers=headers) as response:
            data = await response.json()
            await log_response_details(response, data)

            if response.status != 200:
                logger.error(
                    f"Failed to get metrics. Status: {response.status}, Response: {data}"
                )
                return False

            logger.success("✓ Retrieved metrics successfully")
            return True
    except Exception as e:
        logger.exception(f"Exception in metrics retrieval: {str(e)}")
        return False


async def run_tests():
    """Run all API tests."""
    logger.info("Starting API test suite...")
    logger.info(f"Using base URL: {BASE_URL}")

    timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # Create test user
            user_data = await test_create_user(session)
            if not user_data:
                logger.error("User creation failed, stopping tests.")
                return

            logger.info(
                "User created successfully, proceeding with agent tests..."
            )
            user_data["user_id"]
            api_key = user_data["api_key"]

            # Create test agent
            agent_id = await test_create_agent(session, api_key)
            if not agent_id:
                logger.error("Agent creation failed, stopping tests.")
                return

            logger.info(
                "Agent created successfully, proceeding with other tests..."
            )

            # Run remaining tests
            test_results = []

            # Test metrics retrieval
            logger.info("Testing metrics retrieval...")
            metrics_result = await test_get_metrics(
                session, agent_id, api_key
            )
            test_results.append(("Metrics", metrics_result))

            # Test agent update
            logger.info("Testing agent update...")
            update_result = await test_agent_update(
                session, agent_id, api_key
            )
            test_results.append(("Agent Update", update_result))

            # Test completion
            logger.info("Testing completion...")
            completion_result = await test_completion(
                session, agent_id, api_key
            )
            test_results.append(("Completion", completion_result))

            # Log final results
            logger.info("\nTest Results Summary:")
            all_passed = True
            for test_name, result in test_results:
                status = "PASSED" if result else "FAILED"
                logger.info(f"{test_name}: {status}")
                if not result:
                    all_passed = False

            if all_passed:
                logger.success(
                    "\n🎉 All tests completed successfully!"
                )
            else:
                logger.error(
                    "\n❌ Some tests failed. Check the logs for details."
                )

            logger.info(
                f"\nDetailed logs available at: {os.path.abspath(LOG_PATH)}"
            )

        except Exception as e:
            logger.exception(
                f"Unexpected error during test execution: {str(e)}"
            )
            raise
        finally:
            logger.info("Test suite execution completed.")


def main():
    logger.info("=" * 50)
    logger.info("API TEST SUITE EXECUTION")
    logger.info("=" * 50)

    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        logger.warning("Test execution interrupted by user.")
    except Exception:
        logger.exception("Fatal error in test execution:")
    finally:
        logger.info("Test suite shutdown complete.")


if __name__ == "__main__":
    main()
