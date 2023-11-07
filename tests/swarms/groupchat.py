import pytest

from swarms.models import OpenAIChat
from swarms.structs.flow import Flow, stop_when_repeats

# Mocks and Fixtures
@pytest.fixture
def mocked_llm():
    return OpenAIChat(
        openai_api_key="skkssk",
    )


@pytest.fixture
def basic_flow(mocked_llm):
    return Flow(llm=mocked_llm, max_loops=5)

@pytest.fixture
def basic_flow(mocked_llm):
    return Flow(llm=mocked_llm, max_loops=5)
import pytest
from unittest.mock import MagicMock
from swarms.swarms.groupchat import GroupChat


@pytest.fixture
def mock_flow():
    flow = MagicMock()
    flow.name = 'MockAgent'
    flow.system_message = 'System Message'
    flow.generate_reply = MagicMock(return_value={'role': 'MockAgent', 'content': 'Mock Reply'})
    return flow

@pytest.fixture
def group_chat(mock_flow):
    return GroupChat(agents=[mock_flow, mock_flow], messages=[])

def test_agent_names(group_chat):
    assert group_chat.agent_names == ['MockAgent', 'MockAgent']

def test_reset(group_chat):
    group_chat.messages.append({'role': 'test', 'content': 'test message'})
    group_chat.reset()
    assert not group_chat.messages

def test_agent_by_name_not_found(group_chat):
    with pytest.raises(ValueError):
        group_chat.agent_by_name('NonExistentAgent')

@pytest.mark.parametrize("name", ['MockAgent', 'Agent'])
def test_agent_by_name_found(group_chat, name):
    agent = group_chat.agent_by_name(name)
    assert agent.name == 'MockAgent'




def test_provide_feedback(basic_flow):
    feedback = "Test feedback"
    basic_flow.provide_feedback(feedback)
    assert feedback in basic_flow.feedback
