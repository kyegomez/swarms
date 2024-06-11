import os
import pytest
from dotenv import load_dotenv
from weather_swarm import Agent
from weather_swarm.prompts import (
    WEATHER_AGENT_SYSTEM_PROMPT,
    GLOSSARY_PROMPTS,
    FEW_SHORT_PROMPTS,
)
from weather_swarm.tools.tools import (
    point_query,
    request_ndfd_basic,
    request_ndfd_hourly,
)
from swarms import OpenAIChat
from unittest.mock import Mock, patch

# Load environment variables for tests
load_dotenv()


# Fixtures
@pytest.fixture
def weather_agent():
    return Agent(
        agent_name="WeatherMan Agent",
        system_prompt=WEATHER_AGENT_SYSTEM_PROMPT,
        sop_list=[GLOSSARY_PROMPTS, FEW_SHORT_PROMPTS],
        llm=OpenAIChat(),
        max_loops=1,
        dynamic_temperature_enabled=True,
        verbose=True,
        output_type=str,
        tools=[point_query, request_ndfd_basic, request_ndfd_hourly],
        docs_folder="datasets",
        metadata="json",
        function_calling_format_type="OpenAI",
        function_calling_type="json",
    )


# Test Environment Loading
def test_load_dotenv():
    assert (
        "API_KEY" in os.environ
    ), "API_KEY not found in environment variables"
    assert (
        "API_SECRET" in os.environ
    ), "API_SECRET not found in environment variables"


# Test Agent Initialization
def test_agent_initialization(weather_agent):
    assert weather_agent.agent_name == "WeatherMan Agent"
    assert weather_agent.system_prompt == WEATHER_AGENT_SYSTEM_PROMPT
    assert weather_agent.llm is not None
    assert len(weather_agent.tools) == 3
    assert weather_agent.max_loops == 1
    assert weather_agent.dynamic_temperature_enabled is True
    assert weather_agent.verbose is True
    assert weather_agent.output_type == str
    assert weather_agent.docs_folder == "datasets"
    assert weather_agent.metadata == "json"
    assert weather_agent.function_calling_format_type == "OpenAI"
    assert weather_agent.function_calling_type == "json"


# Parameterized Testing for Agent Tools
@pytest.mark.parametrize(
    "tool", [point_query, request_ndfd_basic, request_ndfd_hourly]
)
def test_agent_tools(weather_agent, tool):
    assert tool in weather_agent.tools


# Mocking the Agent Run Method
@patch.object(
    Agent,
    "run",
    return_value="No, there are no chances of rain today in Huntsville.",
)
def test_agent_run(mock_run, weather_agent):
    response = weather_agent.run(
        "Are there any chances of rain today in Huntsville?"
    )
    assert (
        response
        == "No, there are no chances of rain today in Huntsville."
    )
    mock_run.assert_called_once_with(
        "Are there any chances of rain today in Huntsville?"
    )


# Testing Agent's Response Handling
def test_agent_response_handling(weather_agent):
    weather_agent.llm = Mock()
    weather_agent.llm.return_value = "Mocked Response"
    response = weather_agent.run("What's the weather like?")
    assert response == "Mocked Response"


# Test for Exception Handling in Agent Run
def test_agent_run_exception_handling(weather_agent):
    weather_agent.llm = Mock(
        side_effect=Exception("Mocked Exception")
    )
    with pytest.raises(Exception, match="Mocked Exception"):
        weather_agent.run("Will it rain tomorrow?")


# Testing Agent Initialization with Missing Parameters
def test_agent_initialization_missing_params():
    with pytest.raises(TypeError):
        Agent(agent_name="WeatherMan Agent")


# Mocking Environment Variables
@patch.dict(
    os.environ,
    {"API_KEY": "mock_api_key", "API_SECRET": "mock_api_secret"},
)
def test_environment_variables():
    load_dotenv()
    assert os.getenv("API_KEY") == "mock_api_key"
    assert os.getenv("API_SECRET") == "mock_api_secret"


# Testing Tools Functionality (Example: point_query)
def test_point_query():
    response = point_query("test_latitude", "test_longitude")
    assert (
        response is not None
    )  # Replace with more specific assertions based on actual function behavior


# Testing Tools Functionality (Example: request_ndfd_basic)
def test_request_ndfd_basic():
    response = request_ndfd_basic("test_latitude", "test_longitude")
    assert (
        response is not None
    )  # Replace with more specific assertions based on actual function behavior


# Testing Tools Functionality (Example: request_ndfd_hourly)
def test_request_ndfd_hourly():
    response = request_ndfd_hourly("test_latitude", "test_longitude")
    assert (
        response is not None
    )  # Replace with more specific assertions based on actual function behavior


# Grouping and Marking Tests
@pytest.mark.slow
def test_slow_functionality(weather_agent):
    response = weather_agent.run("Long running query")
    assert response is not None  # Example placeholder


# Test Coverage Report
# Run the following command to generate a coverage report: `pytest --cov=weather_swarm`
