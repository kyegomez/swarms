import json
import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

from swarms.models import OpenAIChat
from swarms.structs.flow import Flow, stop_when_repeats

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


# Mocks and Fixtures
@pytest.fixture
def mocked_llm():
    return OpenAIChat(
        openai_api_key=openai_api_key,
    )


@pytest.fixture
def basic_flow(mocked_llm):
    return Flow(llm=mocked_llm, max_loops=5)


@pytest.fixture
def flow_with_condition(mocked_llm):
    return Flow(llm=mocked_llm, max_loops=5, stopping_condition=stop_when_repeats)


# Basic Tests
def test_stop_when_repeats():
    assert stop_when_repeats("Please Stop now")
    assert not stop_when_repeats("Continue the process")


def test_flow_initialization(basic_flow):
    assert basic_flow.max_loops == 5
    assert basic_flow.stopping_condition is None
    assert basic_flow.loop_interval == 1
    assert basic_flow.retry_attempts == 3
    assert basic_flow.retry_interval == 1
    assert basic_flow.feedback == []
    assert basic_flow.memory == []
    assert basic_flow.task is None
    assert basic_flow.stopping_token == "<DONE>"
    assert not basic_flow.interactive


def test_provide_feedback(basic_flow):
    feedback = "Test feedback"
    basic_flow.provide_feedback(feedback)
    assert feedback in basic_flow.feedback


@patch("time.sleep", return_value=None)  # to speed up tests
def test_run_without_stopping_condition(mocked_sleep, basic_flow):
    response = basic_flow.run("Test task")
    assert response == "Test task"  # since our mocked llm doesn't modify the response


@patch("time.sleep", return_value=None)  # to speed up tests
def test_run_with_stopping_condition(mocked_sleep, flow_with_condition):
    response = flow_with_condition.run("Stop")
    assert response == "Stop"


@patch("time.sleep", return_value=None)  # to speed up tests
def test_run_with_exception(mocked_sleep, basic_flow):
    basic_flow.llm.side_effect = Exception("Test Exception")
    with pytest.raises(Exception, match="Test Exception"):
        basic_flow.run("Test task")


def test_bulk_run(basic_flow):
    inputs = [{"task": "Test1"}, {"task": "Test2"}]
    responses = basic_flow.bulk_run(inputs)
    assert responses == ["Test1", "Test2"]


# Tests involving file IO
def test_save_and_load(basic_flow, tmp_path):
    file_path = tmp_path / "memory.json"
    basic_flow.memory.append(["Test1", "Test2"])
    basic_flow.save(file_path)

    new_flow = Flow(llm=mocked_llm, max_loops=5)
    new_flow.load(file_path)
    assert new_flow.memory == [["Test1", "Test2"]]


# Environment variable mock test
def test_env_variable_handling(monkeypatch):
    monkeypatch.setenv("API_KEY", "test_key")
    assert os.getenv("API_KEY") == "test_key"


# TODO: Add more tests, especially edge cases and exception cases. Implement parametrized tests for varied inputs.


# Test initializing the flow with different stopping conditions
def test_flow_with_custom_stopping_condition(mocked_llm):
    def stopping_condition(x):
        return "terminate" in x.lower()

    flow = Flow(llm=mocked_llm, max_loops=5, stopping_condition=stopping_condition)
    assert flow.stopping_condition("Please terminate now")
    assert not flow.stopping_condition("Continue the process")


# Test calling the flow directly
def test_flow_call(basic_flow):
    response = basic_flow("Test call")
    assert response == "Test call"


# Test formatting the prompt
def test_format_prompt(basic_flow):
    formatted_prompt = basic_flow.format_prompt("Hello {name}", name="John")
    assert formatted_prompt == "Hello John"


# Test with max loops
@patch("time.sleep", return_value=None)
def test_max_loops(mocked_sleep, basic_flow):
    basic_flow.max_loops = 3
    response = basic_flow.run("Looping")
    assert response == "Looping"


# Test stopping token
@patch("time.sleep", return_value=None)
def test_stopping_token(mocked_sleep, basic_flow):
    basic_flow.stopping_token = "Terminate"
    response = basic_flow.run("Loop until Terminate")
    assert response == "Loop until Terminate"


# Test interactive mode
def test_interactive_mode(basic_flow):
    basic_flow.interactive = True
    assert basic_flow.interactive


# Test bulk run with varied inputs
def test_bulk_run_varied_inputs(basic_flow):
    inputs = [{"task": "Test1"}, {"task": "Test2"}, {"task": "Stop now"}]
    responses = basic_flow.bulk_run(inputs)
    assert responses == ["Test1", "Test2", "Stop now"]


# Test loading non-existent file
def test_load_non_existent_file(basic_flow, tmp_path):
    file_path = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        basic_flow.load(file_path)


# Test saving with different memory data
def test_save_different_memory(basic_flow, tmp_path):
    file_path = tmp_path / "memory.json"
    basic_flow.memory.append(["Task1", "Task2", "Task3"])
    basic_flow.save(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)
    assert data == [["Task1", "Task2", "Task3"]]


# Test the stopping condition check
def test_check_stopping_condition(flow_with_condition):
    assert flow_with_condition._check_stopping_condition("Stop this process")
    assert not flow_with_condition._check_stopping_condition("Continue the task")


# Test without providing max loops (default value should be 5)
def test_default_max_loops(mocked_llm):
    flow = Flow(llm=mocked_llm)
    assert flow.max_loops == 5


# Test creating flow from llm and template
def test_from_llm_and_template(mocked_llm):
    flow = Flow.from_llm_and_template(mocked_llm, "Test template")
    assert isinstance(flow, Flow)


# Mocking the OpenAIChat for testing
@patch("swarms.models.OpenAIChat", autospec=True)
def test_mocked_openai_chat(MockedOpenAIChat):
    llm = MockedOpenAIChat(openai_api_key=openai_api_key)
    llm.return_value = MagicMock()
    flow = Flow(llm=llm, max_loops=5)
    flow.run("Mocked run")
    assert MockedOpenAIChat.called


# Test retry attempts
@patch("time.sleep", return_value=None)
def test_retry_attempts(mocked_sleep, basic_flow):
    basic_flow.retry_attempts = 2
    basic_flow.llm.side_effect = [Exception("Test Exception"), "Valid response"]
    response = basic_flow.run("Test retry")
    assert response == "Valid response"


# Test different loop intervals
@patch("time.sleep", return_value=None)
def test_different_loop_intervals(mocked_sleep, basic_flow):
    basic_flow.loop_interval = 2
    response = basic_flow.run("Test loop interval")
    assert response == "Test loop interval"


# Test different retry intervals
@patch("time.sleep", return_value=None)
def test_different_retry_intervals(mocked_sleep, basic_flow):
    basic_flow.retry_interval = 2
    response = basic_flow.run("Test retry interval")
    assert response == "Test retry interval"


# Test invoking the flow with additional kwargs
@patch("time.sleep", return_value=None)
def test_flow_call_with_kwargs(mocked_sleep, basic_flow):
    response = basic_flow("Test call", param1="value1", param2="value2")
    assert response == "Test call"


# Test initializing the flow with all parameters
def test_flow_initialization_all_params(mocked_llm):
    flow = Flow(
        llm=mocked_llm,
        max_loops=10,
        stopping_condition=stop_when_repeats,
        loop_interval=2,
        retry_attempts=4,
        retry_interval=2,
        interactive=True,
        param1="value1",
        param2="value2",
    )
    assert flow.max_loops == 10
    assert flow.loop_interval == 2
    assert flow.retry_attempts == 4
    assert flow.retry_interval == 2
    assert flow.interactive


# Test the stopping token is in the response
@patch("time.sleep", return_value=None)
def test_stopping_token_in_response(mocked_sleep, basic_flow):
    response = basic_flow.run("Test stopping token")
    assert basic_flow.stopping_token in response
