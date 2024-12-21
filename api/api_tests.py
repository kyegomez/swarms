import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from loguru import logger

# Configure logger
logger.add(
    "tests/api_test_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


class TestConfig:
    """Test configuration and utilities"""

    BASE_URL: str = "http://localhost:8000/v1"
    TEST_USERNAME: str = "test_user"
    api_key: Optional[str] = None
    user_id: Optional[UUID] = None
    test_agent_id: Optional[UUID] = None


class TestResult:
    """Model for test results"""

    def __init__(
        self,
        test_name: str,
        status: str,
        duration: float,
        error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.test_name = test_name
        self.status = status
        self.duration = duration
        self.error = error
        self.details = details or {}

    def dict(self):
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration": self.duration,
            "error": self.error,
            "details": self.details,
        }


async def log_response(
    response: httpx.Response, test_name: str
) -> None:
    """Log API response details"""
    logger.debug(f"\n{test_name} Response:")
    logger.debug(f"Status Code: {response.status_code}")
    logger.debug(f"Headers: {dict(response.headers)}")
    try:
        logger.debug(f"Body: {response.json()}")
    except json.JSONDecodeError:
        logger.debug(f"Body: {response.text}")


async def create_test_user() -> TestResult:
    """Create a test user and get API key"""
    start_time = datetime.now()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TestConfig.BASE_URL}/users",
                json={"username": TestConfig.TEST_USERNAME},
            )
            await log_response(response, "Create User")

            if response.status_code == 200:
                data = response.json()
                TestConfig.api_key = data["api_key"]
                TestConfig.user_id = UUID(data["user_id"])
                return TestResult(
                    test_name="create_test_user",
                    status="passed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    details={"user_id": str(TestConfig.user_id)},
                )
            else:
                return TestResult(
                    test_name="create_test_user",
                    status="failed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    error=f"Failed to create user: {response.text}",
                )
    except Exception as e:
        logger.error(f"Error in create_test_user: {str(e)}")
        return TestResult(
            test_name="create_test_user",
            status="error",
            duration=(datetime.now() - start_time).total_seconds(),
            error=str(e),
        )


async def create_test_agent() -> TestResult:
    """Create a test agent"""
    start_time = datetime.now()
    try:
        # Create agent config according to the AgentConfig model
        agent_config = {
            "agent_name": "test_agent",
            "model_name": "gpt-4",
            "description": "Test agent for API testing",
            "system_prompt": "You are a test agent.",
            "temperature": 0.1,
            "max_loops": 1,
            "dynamic_temperature_enabled": True,
            "user_name": TestConfig.TEST_USERNAME,
            "retry_attempts": 1,
            "context_length": 4000,
            "output_type": "string",
            "streaming_on": False,
            "tags": ["test", "api"],
            "stopping_token": "<DONE>",
            "auto_generate_prompt": False,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TestConfig.BASE_URL}/agent",
                json=agent_config,
                headers={"api-key": TestConfig.api_key},
            )
            await log_response(response, "Create Agent")

            if response.status_code == 200:
                data = response.json()
                TestConfig.test_agent_id = UUID(data["agent_id"])
                return TestResult(
                    test_name="create_test_agent",
                    status="passed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    details={
                        "agent_id": str(TestConfig.test_agent_id)
                    },
                )
            else:
                return TestResult(
                    test_name="create_test_agent",
                    status="failed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    error=f"Failed to create agent: {response.text}",
                )
    except Exception as e:
        logger.error(f"Error in create_test_agent: {str(e)}")
        return TestResult(
            test_name="create_test_agent",
            status="error",
            duration=(datetime.now() - start_time).total_seconds(),
            error=str(e),
        )


async def test_agent_completion() -> TestResult:
    """Test agent completion endpoint"""
    start_time = datetime.now()
    try:
        completion_request = {
            "prompt": "Hello, this is a test prompt.",
            "agent_id": str(TestConfig.test_agent_id),
            "max_tokens": 100,
            "temperature_override": 0.5,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TestConfig.BASE_URL}/agent/completions",
                json=completion_request,
                headers={"api-key": TestConfig.api_key},
            )
            await log_response(response, "Agent Completion")

            if response.status_code == 200:
                return TestResult(
                    test_name="test_agent_completion",
                    status="passed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    details={"response": response.json()},
                )
            else:
                return TestResult(
                    test_name="test_agent_completion",
                    status="failed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    error=f"Failed completion test: {response.text}",
                )
    except Exception as e:
        logger.error(f"Error in test_agent_completion: {str(e)}")
        return TestResult(
            test_name="test_agent_completion",
            status="error",
            duration=(datetime.now() - start_time).total_seconds(),
            error=str(e),
        )


async def test_agent_metrics() -> TestResult:
    """Test agent metrics endpoint"""
    start_time = datetime.now()
    try:
        if not TestConfig.test_agent_id:
            return TestResult(
                test_name="test_agent_metrics",
                status="failed",
                duration=(
                    datetime.now() - start_time
                ).total_seconds(),
                error="No test agent ID available",
            )

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TestConfig.BASE_URL}/agent/{str(TestConfig.test_agent_id)}/metrics",
                headers={"api-key": TestConfig.api_key},
            )
            await log_response(response, "Agent Metrics")

            if response.status_code == 200:
                return TestResult(
                    test_name="test_agent_metrics",
                    status="passed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    details={"metrics": response.json()},
                )
            else:
                return TestResult(
                    test_name="test_agent_metrics",
                    status="failed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    error=f"Failed metrics test: {response.text}",
                )
    except Exception as e:
        logger.error(f"Error in test_agent_metrics: {str(e)}")
        return TestResult(
            test_name="test_agent_metrics",
            status="error",
            duration=(datetime.now() - start_time).total_seconds(),
            error=str(e),
        )


async def test_update_agent() -> TestResult:
    """Test agent update endpoint"""
    start_time = datetime.now()
    try:
        if not TestConfig.test_agent_id:
            return TestResult(
                test_name="test_update_agent",
                status="failed",
                duration=(
                    datetime.now() - start_time
                ).total_seconds(),
                error="No test agent ID available",
            )

        update_data = {
            "description": "Updated test agent description",
            "tags": ["test", "updated"],
            "max_loops": 2,
        }

        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{TestConfig.BASE_URL}/agent/{str(TestConfig.test_agent_id)}",
                json=update_data,
                headers={"api-key": TestConfig.api_key},
            )
            await log_response(response, "Update Agent")

            if response.status_code == 200:
                return TestResult(
                    test_name="test_update_agent",
                    status="passed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    details={"update_response": response.json()},
                )
            else:
                return TestResult(
                    test_name="test_update_agent",
                    status="failed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    error=f"Failed update test: {response.text}",
                )
    except Exception as e:
        logger.error(f"Error in test_update_agent: {str(e)}")
        return TestResult(
            test_name="test_update_agent",
            status="error",
            duration=(datetime.now() - start_time).total_seconds(),
            error=str(e),
        )


async def test_error_handling() -> TestResult:
    """Test API error handling"""
    start_time = datetime.now()
    try:
        async with httpx.AsyncClient() as client:
            # Test with invalid API key
            invalid_agent_id = "00000000-0000-0000-0000-000000000000"
            response = await client.get(
                f"{TestConfig.BASE_URL}/agent/{invalid_agent_id}/metrics",
                headers={"api-key": "invalid_key"},
            )
            await log_response(response, "Invalid API Key Test")

            if response.status_code in [401, 403]:
                return TestResult(
                    test_name="test_error_handling",
                    status="passed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    details={"error_response": response.json()},
                )
            else:
                return TestResult(
                    test_name="test_error_handling",
                    status="failed",
                    duration=(
                        datetime.now() - start_time
                    ).total_seconds(),
                    error="Error handling test failed",
                )
    except Exception as e:
        logger.error(f"Error in test_error_handling: {str(e)}")
        return TestResult(
            test_name="test_error_handling",
            status="error",
            duration=(datetime.now() - start_time).total_seconds(),
            error=str(e),
        )


async def cleanup_test_resources() -> TestResult:
    """Clean up test resources"""
    start_time = datetime.now()
    try:
        if TestConfig.test_agent_id:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{TestConfig.BASE_URL}/agent/{str(TestConfig.test_agent_id)}",
                    headers={"api-key": TestConfig.api_key},
                )
                await log_response(response, "Delete Agent")

        return TestResult(
            test_name="cleanup_test_resources",
            status="passed",
            duration=(datetime.now() - start_time).total_seconds(),
            details={"cleanup": "completed"},
        )
    except Exception as e:
        logger.error(f"Error in cleanup_test_resources: {str(e)}")
        return TestResult(
            test_name="cleanup_test_resources",
            status="error",
            duration=(datetime.now() - start_time).total_seconds(),
            error=str(e),
        )


async def run_all_tests() -> List[TestResult]:
    """Run all tests in sequence"""
    logger.info("Starting API test suite")
    results = []

    # Initialize
    results.append(await create_test_user())
    if results[-1].status != "passed":
        logger.error(
            "Failed to create test user, aborting remaining tests"
        )
        return results

    # Add delay to ensure user is properly created
    await asyncio.sleep(1)

    # Core tests
    test_functions = [
        create_test_agent,
        test_agent_completion,
        test_agent_metrics,
        test_update_agent,
        test_error_handling,
    ]

    for test_func in test_functions:
        result = await test_func()
        results.append(result)
        logger.info(f"Test {result.test_name}: {result.status}")
        if result.error:
            logger.error(
                f"Error in {result.test_name}: {result.error}"
            )

        # Add small delay between tests
        await asyncio.sleep(0.5)

    # Cleanup
    results.append(await cleanup_test_resources())

    # Log summary
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    errors = sum(1 for r in results if r.status == "error")

    logger.info("\nTest Summary:")
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Errors: {errors}")

    return results


def main():
    """Main entry point for running tests"""
    logger.info("Starting API testing suite")
    try:
        results = asyncio.run(run_all_tests())

        # Write results to JSON file
        with open("test_results.json", "w") as f:
            json.dump(
                [result.dict() for result in results],
                f,
                indent=2,
                default=str,
            )

        logger.info("Test results written to test_results.json")

    except Exception:
        logger.error("Fatal error in test suite: ")


main()
