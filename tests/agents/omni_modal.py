import pytest
from langchain.base_language import BaseLanguageModel

from swarms.agents.omni_modal_agent import (
    OmniModalAgent,  # Replace `your_module_name` with the appropriate module name
)


# Mock objects or set up fixtures for dependent classes or external methods
@pytest.fixture
def mock_llm():
    # For this mock, we are assuming the BaseLanguageModel has a method named "process"
    class MockLLM(BaseLanguageModel):
        def process(self, input):
            return "mock response"

    return MockLLM()


@pytest.fixture
def omni_agent(mock_llm):
    return OmniModalAgent(mock_llm)


def test_omnimodalagent_initialization(omni_agent):
    assert omni_agent.llm is not None, "LLM initialization failed"
    assert len(omni_agent.tools) > 0, "Tools initialization failed"


def test_omnimodalagent_run(omni_agent):
    input_string = "Hello, how are you?"
    response = omni_agent.run(input_string)
    assert response is not None, "Response generation failed"
    assert isinstance(response, str), "Response should be a string"


def test_task_executor_initialization(omni_agent):
    assert omni_agent.task_executor is not None, "TaskExecutor initialization failed"
