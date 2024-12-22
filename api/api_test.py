import os
from typing import Dict, Optional, Any
from dataclasses import dataclass
import pytest
import requests
from uuid import UUID
from pydantic import BaseModel
from _pytest.terminal import TerminalReporter


# Configuration
@dataclass
class TestConfig:
    """Test configuration settings"""

    base_url: str
    timeout: int = 30
    verify_ssl: bool = True


# Load config from environment or use defaults
config = TestConfig(
    base_url=os.getenv("API_BASE_URL", "http://localhost:8000/v1")
)


# API Response Types
class UserResponse(BaseModel):
    user_id: str
    api_key: str


class AgentResponse(BaseModel):
    agent_id: UUID


class MetricsResponse(BaseModel):
    total_completions: int
    average_response_time: float
    error_rate: float
    last_24h_completions: int
    total_tokens_used: int
    uptime_percentage: float
    success_rate: float
    peak_tokens_per_minute: int


class APIClient:
    """API Client with typed methods"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        """Construct full URL"""
        return f"{self.config.base_url}/{path.lstrip('/')}"

    def _request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Make HTTP request with config defaults"""
        url = self._url(path)
        return self.session.request(
            method=method,
            url=url,
            headers=headers,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
            **kwargs,
        )

    def create_user(self, username: str) -> UserResponse:
        """Create a new user"""
        response = self._request(
            "POST", "/users", json={"username": username}
        )
        response.raise_for_status()
        return UserResponse(**response.json())

    def create_agent(
        self, agent_config: Dict[str, Any], api_key: str
    ) -> AgentResponse:
        """Create a new agent"""
        headers = {"api-key": api_key}
        response = self._request(
            "POST", "/agent", headers=headers, json=agent_config
        )
        response.raise_for_status()
        return AgentResponse(**response.json())

    def get_metrics(
        self, agent_id: UUID, api_key: str
    ) -> MetricsResponse:
        """Get agent metrics"""
        headers = {"api-key": api_key}
        response = self._request(
            "GET", f"/agent/{agent_id}/metrics", headers=headers
        )
        response.raise_for_status()
        return MetricsResponse(**response.json())


# Test Fixtures
@pytest.fixture
def api_client() -> APIClient:
    """Fixture for API client"""
    return APIClient(config)


@pytest.fixture
def test_user(api_client: APIClient) -> UserResponse:
    """Fixture for test user"""
    return api_client.create_user("test_user")


@pytest.fixture
def test_agent(
    api_client: APIClient, test_user: UserResponse
) -> AgentResponse:
    """Fixture for test agent"""
    agent_config = {
        "agent_name": "test_agent",
        "model_name": "gpt-4",
        "system_prompt": "You are a test agent",
        "description": "Test agent description",
    }
    return api_client.create_agent(agent_config, test_user.api_key)


# Tests
def test_user_creation(api_client: APIClient):
    """Test user creation flow"""
    response = api_client.create_user("new_test_user")
    assert response.user_id
    assert response.api_key


def test_agent_creation(
    api_client: APIClient, test_user: UserResponse
):
    """Test agent creation flow"""
    agent_config = {
        "agent_name": "test_agent",
        "model_name": "gpt-4",
        "system_prompt": "You are a test agent",
        "description": "Test agent description",
    }
    response = api_client.create_agent(
        agent_config, test_user.api_key
    )
    assert response.agent_id


def test_agent_metrics(
    api_client: APIClient,
    test_user: UserResponse,
    test_agent: AgentResponse,
):
    """Test metrics retrieval"""
    metrics = api_client.get_metrics(
        test_agent.agent_id, test_user.api_key
    )
    assert metrics.total_completions >= 0
    assert metrics.error_rate >= 0
    assert metrics.uptime_percentage >= 0


def test_invalid_auth(api_client: APIClient):
    """Test invalid authentication"""
    with pytest.raises(requests.exceptions.HTTPError) as exc_info:
        api_client.create_agent({}, "invalid_key")
    assert exc_info.value.response.status_code == 401


# Custom pytest plugin to capture test results
class ResultCapture:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0


@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(
    terminalreporter: TerminalReporter, exitstatus: int
):
    yield
    capture = getattr(
        terminalreporter.config, "_result_capture", None
    )
    if capture:
        capture.total = (
            len(terminalreporter.stats.get("passed", []))
            + len(terminalreporter.stats.get("failed", []))
            + len(terminalreporter.stats.get("error", []))
        )
        capture.passed = len(terminalreporter.stats.get("passed", []))
        capture.failed = len(terminalreporter.stats.get("failed", []))
        capture.errors = len(terminalreporter.stats.get("error", []))


@dataclass
class TestReport:
    total_tests: int
    passed: int
    failed: int
    errors: int

    @property
    def success_rate(self) -> float:
        return (
            (self.passed / self.total_tests) * 100
            if self.total_tests > 0
            else 0
        )


def run_tests() -> TestReport:
    """Run tests and generate typed report"""
    # Create result capture
    capture = ResultCapture()

    # Create pytest configuration
    args = [__file__, "-v"]

    # Run pytest with our plugin
    pytest.main(args, plugins=[capture])

    # Generate report
    return TestReport(
        total_tests=capture.total,
        passed=capture.passed,
        failed=capture.failed,
        errors=capture.errors,
    )


if __name__ == "__main__":
    # Example usage with environment variable
    # export API_BASE_URL=http://api.example.com/v1

    report = run_tests()
    print("\nTest Results:")
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed}")
    print(f"Failed: {report.failed}")
    print(f"Errors: {report.errors}")
    print(f"Success Rate: {report.success_rate:.2f}%")
