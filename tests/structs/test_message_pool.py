from swarms import OpenAIChat
from swarms.structs.agent import Agent
from swarms.structs.message_pool import MessagePool


def test_message_pool_initialization():
    agent1 = Agent(llm=OpenAIChat(), agent_name="agent1")
    agent2 = Agent(llm=OpenAIChat(), agent_name="agent1")
    moderator = Agent(llm=OpenAIChat(), agent_name="agent1")
    agents = [agent1, agent2]
    message_pool = MessagePool(
        agents=agents, moderator=moderator, turns=5
    )

    assert message_pool.agent == agents
    assert message_pool.moderator == moderator
    assert message_pool.turns == 5
    assert message_pool.messages == []


def test_message_pool_add():
    agent1 = Agent(llm=OpenAIChat(), agent_name="agent1")
    message_pool = MessagePool(
        agents=[agent1], moderator=agent1, turns=5
    )
    message_pool.add(agent=agent1, content="Hello, world!", turn=1)

    assert message_pool.messages == [
        {
            "agent": agent1,
            "content": "Hello, world!",
            "turn": 1,
            "visible_to": "all",
            "logged": True,
        }
    ]


def test_message_pool_reset():
    agent1 = Agent(llm=OpenAIChat(), agent_name="agent1")
    message_pool = MessagePool(
        agents=[agent1], moderator=agent1, turns=5
    )
    message_pool.add(agent=agent1, content="Hello, world!", turn=1)
    message_pool.reset()

    assert message_pool.messages == []


def test_message_pool_last_turn():
    agent1 = Agent(llm=OpenAIChat(), agent_name="agent1")
    message_pool = MessagePool(
        agents=[agent1], moderator=agent1, turns=5
    )
    message_pool.add(agent=agent1, content="Hello, world!", turn=1)

    assert message_pool.last_turn() == 1


def test_message_pool_last_message():
    agent1 = Agent(llm=OpenAIChat(), agent_name="agent1")
    message_pool = MessagePool(
        agents=[agent1], moderator=agent1, turns=5
    )
    message_pool.add(agent=agent1, content="Hello, world!", turn=1)

    assert message_pool.last_message == {
        "agent": agent1,
        "content": "Hello, world!",
        "turn": 1,
        "visible_to": "all",
        "logged": True,
    }


def test_message_pool_get_all_messages():
    agent1 = Agent(llm=OpenAIChat(), agent_name="agent1")
    message_pool = MessagePool(
        agents=[agent1], moderator=agent1, turns=5
    )
    message_pool.add(agent=agent1, content="Hello, world!", turn=1)

    assert message_pool.get_all_messages() == [
        {
            "agent": agent1,
            "content": "Hello, world!",
            "turn": 1,
            "visible_to": "all",
            "logged": True,
        }
    ]


def test_message_pool_get_visible_messages():
    agent1 = Agent(llm=OpenAIChat(), agent_name="agent1")
    agent2 = Agent(agent_name="agent2")
    message_pool = MessagePool(
        agents=[agent1, agent2], moderator=agent1, turns=5
    )
    message_pool.add(
        agent=agent1,
        content="Hello, agent2!",
        turn=1,
        visible_to=[agent2.agent_name],
    )

    assert message_pool.get_visible_messages(
        agent=agent2, turn=2
    ) == [
        {
            "agent": agent1,
            "content": "Hello, agent2!",
            "turn": 1,
            "visible_to": [agent2.agent_name],
            "logged": True,
        }
    ]
