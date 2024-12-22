import os
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import requests
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api_tests.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Configuration
@dataclass
class TestConfig:
    """Test configuration settings"""

    base_url: str
    timeout: int = 30
    verify_ssl: bool = True
    debug: bool = True


# Load config from environment or use defaults
config = TestConfig(
    base_url=os.getenv("API_BASE_URL", "http://0.0.0.0:8000/v1")
)


class APIClient:
    """API Client for testing"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        """Construct full URL"""
        return f"{self.config.base_url}/{path.lstrip('/')}"

    def _log_request_details(
        self, method: str, url: str, headers: Dict, data: Any
    ):
        """Log request details for debugging"""
        logger.info("\nRequest Details:")
        logger.info(f"Method: {method}")
        logger.info(f"URL: {url}")
        logger.info(f"Headers: {json.dumps(headers, indent=2)}")
        logger.info(
            f"Data: {json.dumps(data, indent=2) if data else None}"
        )

    def _log_response_details(self, response: requests.Response):
        """Log response details for debugging"""
        logger.info("\nResponse Details:")
        logger.info(f"Status Code: {response.status_code}")
        logger.info(
            f"Headers: {json.dumps(dict(response.headers), indent=2)}"
        )
        try:
            logger.info(
                f"Body: {json.dumps(response.json(), indent=2)}"
            )
        except Exception:
            logger.info(f"Body: {response.text}")

    def _request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Make HTTP request with config defaults"""
        url = self._url(path)
        headers = headers or {}

        if self.config.debug:
            self._log_request_details(
                method, url, headers, kwargs.get("json")
            )

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                **kwargs,
            )

            if self.config.debug:
                self._log_response_details(response)

            if response.status_code >= 400:
                logger.error(
                    f"Request failed with status {response.status_code}"
                )
                logger.error(f"Response: {response.text}")

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Error response: {e.response.text}")
            raise


class TestRunner:
    """Test runner with logging and reporting"""

    def __init__(self):
        self.client = APIClient(config)
        self.results = {"passed": 0, "failed": 0, "total_time": 0}
        self.api_key = None
        self.user_id = None
        self.agent_id = None

    def run_test(self, test_name: str, test_func: callable):
        """Run a single test with timing and logging"""
        logger.info(f"\nRunning test: {test_name}")
        start_time = time.time()

        try:
            test_func()
            self.results["passed"] += 1
            logger.info(f"✅ {test_name} - PASSED")
        except Exception as e:
            self.results["failed"] += 1
            logger.error(f"❌ {test_name} - FAILED: {str(e)}")
            logger.exception(e)

        end_time = time.time()
        duration = end_time - start_time
        self.results["total_time"] += duration
        logger.info(f"Test duration: {duration:.2f}s")

    def test_user_creation(self):
        """Test user creation"""
        response = self.client._request(
            "POST", "/users", json={"username": "test_user"}
        )
        data = response.json()
        assert "user_id" in data, "No user_id in response"
        assert "api_key" in data, "No api_key in response"
        self.api_key = data["api_key"]
        self.user_id = data["user_id"]
        logger.info(f"Created user with ID: {self.user_id}")

    def test_create_api_key(self):
        """Test API key creation"""
        headers = {"api-key": self.api_key}
        response = self.client._request(
            "POST",
            f"/users/{self.user_id}/api-keys",
            headers=headers,
            json={"name": "test_key"},
        )
        data = response.json()
        assert "key" in data, "No key in response"
        logger.info("Successfully created new API key")

    def test_create_agent(self):
        """Test agent creation"""
        headers = {"api-key": self.api_key}
        agent_config = {
            "agent_name": "test_agent",
            "model_name": "gpt-4",
            "system_prompt": "You are a test agent",
            "description": "Test agent description",
            "temperature": 0.7,
            "max_loops": 1,
        }
        response = self.client._request(
            "POST", "/agent", headers=headers, json=agent_config
        )
        data = response.json()
        assert "agent_id" in data, "No agent_id in response"
        self.agent_id = data["agent_id"]
        logger.info(f"Created agent with ID: {self.agent_id}")

        # Wait a bit for agent to be ready
        time.sleep(2)

    def test_list_agents(self):
        """Test agent listing"""
        headers = {"api-key": self.api_key}
        response = self.client._request(
            "GET", "/agents", headers=headers
        )
        agents = response.json()
        assert isinstance(agents, list), "Response is not a list"
        assert len(agents) > 0, "No agents returned"
        logger.info(f"Successfully retrieved {len(agents)} agents")

    def test_agent_completion(self):
        """Test agent completion"""
        if not self.agent_id:
            logger.error("No agent_id available for completion test")
            raise ValueError("Agent ID not set")

        headers = {"api-key": self.api_key}
        completion_request = {
            "prompt": "Write 'Hello World!'",
            "agent_id": str(
                self.agent_id
            ),  # Ensure UUID is converted to string
            "max_tokens": 100,
            "stream": False,
            "temperature_override": 0.7,
        }

        logger.info(
            f"Sending completion request for agent {self.agent_id}"
        )
        response = self.client._request(
            "POST",
            "/agent/completions",
            headers=headers,
            json=completion_request,
        )
        data = response.json()
        assert "response" in data, "No response in completion"
        logger.info(f"Completion response: {data.get('response')}")

    def run_all_tests(self):
        """Run all tests and generate report"""
        logger.info("\n" + "=" * 50)
        logger.info("Starting API test suite...")
        logger.info(f"Base URL: {config.base_url}")
        logger.info("=" * 50 + "\n")

        # Define test sequence
        tests = [
            ("User Creation", self.test_user_creation),
            ("API Key Creation", self.test_create_api_key),
            ("Agent Creation", self.test_create_agent),
            ("List Agents", self.test_list_agents),
            ("Agent Completion", self.test_agent_completion),
        ]

        # Run tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)

        # Generate report
        self.print_report()

    def print_report(self):
        """Print test results report"""
        total_tests = self.results["passed"] + self.results["failed"]
        success_rate = (
            (self.results["passed"] / total_tests * 100)
            if total_tests > 0
            else 0
        )

        report = f"""
\n{'='*50}
API TEST RESULTS
{'='*50}
Total Tests: {total_tests}
Passed: {self.results['passed']} ✅
Failed: {self.results['failed']} ❌
Success Rate: {success_rate:.2f}%
Total Time: {self.results['total_time']:.2f}s
{'='*50}
"""
        logger.info(report)


if __name__ == "__main__":
    try:
        runner = TestRunner()
        runner.run_all_tests()
    except KeyboardInterrupt:
        logger.info("\nTest suite interrupted by user")
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        logger.exception(e)
