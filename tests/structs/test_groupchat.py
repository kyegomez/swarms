import pytest

from swarm_models import OpenAIChat
from swarm_models.anthropic import Anthropic
from swarms.structs.agent import Agent
from swarms.structs.groupchat import GroupChat, GroupChatManager

llm = OpenAIChat()
llm2 = Anthropic()


# Mock the OpenAI class for testing
class MockOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    def generate_reply(self, content):
        return {"role": "mocked_agent", "content": "Mocked Reply"}


# Create fixtures for agents and a sample message
@pytest.fixture
def agent1():
    return Agent(name="Agent1", llm=llm)


@pytest.fixture
def agent2():
    return Agent(name="Agent2", llm=llm2)


@pytest.fixture
def sample_message():
    return {"role": "Agent1", "content": "Hello, World!"}


# Test the initialization of GroupChat
def test_groupchat_initialization(agent1, agent2):
    groupchat = GroupChat(agents=[agent1, agent2])
    assert len(groupchat.agents) == 2
    assert len(groupchat.messages) == 0
    assert groupchat.max_round == 10
    assert groupchat.admin_name == "Admin"


# Test resetting the GroupChat
def test_groupchat_reset(agent1, agent2, sample_message):
    groupchat = GroupChat(agents=[agent1, agent2])
    groupchat.messages.append(sample_message)
    groupchat.reset()
    assert len(groupchat.messages) == 0


# Test finding an agent by name
def test_groupchat_find_agent_by_name(agent1, agent2):
    groupchat = GroupChat(agents=[agent1, agent2])
    found_agent = groupchat.agent_by_name("Agent1")
    assert found_agent == agent1


# Test selecting the next agent
def test_groupchat_select_next_agent(agent1, agent2):
    groupchat = GroupChat(agents=[agent1, agent2])
    next_agent = groupchat.next_agent(agent1)
    assert next_agent == agent2


# Add more tests for different methods and scenarios as needed


# Test the GroupChatManager
def test_groupchat_manager(agent1, agent2):
    groupchat = GroupChat(agents=[agent1, agent2])
    selector = agent1  # Assuming agent1 is the selector
    manager = GroupChatManager(groupchat, selector)
    task = "Task for agent2"
    reply = manager(task)
    assert reply["role"] == "Agent2"
    assert reply["content"] == "Reply from Agent2"


# Test selecting the next speaker when there is only one agent
def test_groupchat_select_speaker_single_agent(agent1):
    groupchat = GroupChat(agents=[agent1])
    selector = agent1
    manager = GroupChatManager(groupchat, selector)
    task = "Task for agent1"
    reply = manager(task)
    assert reply["role"] == "Agent1"
    assert reply["content"] == "Reply from Agent1"


# Test selecting the next speaker when GroupChat is underpopulated
def test_groupchat_select_speaker_underpopulated(agent1, agent2):
    groupchat = GroupChat(agents=[agent1, agent2])
    selector = agent1
    manager = GroupChatManager(groupchat, selector)
    task = "Task for agent1"
    reply = manager(task)
    assert reply["role"] == "Agent2"
    assert reply["content"] == "Reply from Agent2"


# Test formatting history
def test_groupchat_format_history(agent1, agent2, sample_message):
    groupchat = GroupChat(agents=[agent1, agent2])
    groupchat.messages.append(sample_message)
    formatted_history = groupchat.format_history(groupchat.messages)
    expected_history = "'Agent1:Hello, World!"
    assert formatted_history == expected_history


# Test agent names property
def test_groupchat_agent_names(agent1, agent2):
    groupchat = GroupChat(agents=[agent1, agent2])
    names = groupchat.agent_names
    assert len(names) == 2
    assert "Agent1" in names
    assert "Agent2" in names


# Test GroupChatManager initialization
def test_groupchat_manager_initialization(agent1, agent2):
    groupchat = GroupChat(agents=[agent1, agent2])
    selector = agent1
    manager = GroupChatManager(groupchat, selector)
    assert manager.groupchat == groupchat
    assert manager.selector == selector


# Test case to ensure GroupChatManager generates a reply from an agent
def test_groupchat_manager_generate_reply():
    # Create a GroupChat with two agents
    agents = [agent1, agent2]
    groupchat = GroupChat(agents=agents, messages=[], max_round=10)

    # Mock the OpenAI class and GroupChat selector
    mocked_openai = MockOpenAI()
    selector = agent1

    # Initialize GroupChatManager
    manager = GroupChatManager(
        groupchat=groupchat, selector=selector, openai=mocked_openai
    )

    # Generate a reply
    task = "Write me a riddle"
    reply = manager(task)

    # Check if a valid reply is generated
    assert "role" in reply
    assert "content" in reply
    assert reply["role"] in groupchat.agent_names


# Test case to ensure GroupChat selects the next speaker correctly
def test_groupchat_select_speaker():
    agent3 = Agent(name="agent3", llm=llm)
    agents = [agent1, agent2, agent3]
    groupchat = GroupChat(agents=agents, messages=[], max_round=10)

    # Initialize GroupChatManager with agent1 as selector
    selector = agent1
    manager = GroupChatManager(groupchat=groupchat, selector=selector)

    # Simulate selecting the next speaker
    last_speaker = agent1
    next_speaker = manager.select_speaker(
        last_speaker=last_speaker, selector=selector
    )

    # Ensure the next speaker is agent2
    assert next_speaker == agent2


# Test case to ensure GroupChat handles underpopulated group correctly
def test_groupchat_underpopulated_group():
    agent1 = Agent(name="agent1", llm=llm)
    agents = [agent1]
    groupchat = GroupChat(agents=agents, messages=[], max_round=10)

    # Initialize GroupChatManager with agent1 as selector
    selector = agent1
    manager = GroupChatManager(groupchat=groupchat, selector=selector)

    # Simulate selecting the next speaker in an underpopulated group
    last_speaker = agent1
    next_speaker = manager.select_speaker(
        last_speaker=last_speaker, selector=selector
    )

    # Ensure the next speaker is the same as the last speaker in an underpopulated group
    assert next_speaker == last_speaker


# Test case to ensure GroupChatManager handles the maximum rounds correctly
def test_groupchat_max_rounds():
    agents = [agent1, agent2]
    groupchat = GroupChat(agents=agents, messages=[], max_round=2)

    # Initialize GroupChatManager with agent1 as selector
    selector = agent1
    manager = GroupChatManager(groupchat=groupchat, selector=selector)

    # Simulate the conversation with max rounds
    last_speaker = agent1
    for _ in range(2):
        next_speaker = manager.select_speaker(
            last_speaker=last_speaker, selector=selector
        )
        last_speaker = next_speaker

    # Try one more round, should stay with the last speaker
    next_speaker = manager.select_speaker(
        last_speaker=last_speaker, selector=selector
    )

    # Ensure the next speaker is the same as the last speaker after reaching max rounds
    assert next_speaker == last_speaker


# Continue adding more test cases as needed to cover various scenarios and functionalities of the code.
