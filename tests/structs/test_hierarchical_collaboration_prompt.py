from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.ma_utils import list_all_agents


class DummyWorker:
    def __init__(self, name, description=None, system_prompt=None):
        self.agent_name = name
        self.description = description
        self.system_prompt = system_prompt or f"System prompt for {name}"


def test_hierarchical_swarm_add_collaboration_prompt_true():
    """When add_collaboration_prompt=True, director conversation contains collaboration principles."""
    w1 = DummyWorker("Analyst", "Analyses data")
    w2 = DummyWorker("Writer", "Writes reports")

    swarm = HierarchicalSwarm(
        agents=[w1, w2],
        add_collaboration_prompt=True,
        print_on=False,
    )
    conv_text = swarm.conversation.get_str()
    assert "Fundamental Collaboration Principles" in conv_text
    assert "Total Agents: 2" in conv_text
    assert "Analyst" in conv_text
    assert "Writer" in conv_text


def test_hierarchical_swarm_add_collaboration_prompt_false():
    """When add_collaboration_prompt=False, director conversation has roster but omits collaboration principles."""
    w1 = DummyWorker("Analyst", "Analyses data")
    w2 = DummyWorker("Writer", "Writes reports")

    swarm = HierarchicalSwarm(
        agents=[w1, w2],
        add_collaboration_prompt=False,
        print_on=False,
    )
    conv_text = swarm.conversation.get_str()
    assert "Fundamental Collaboration Principles" not in conv_text
    assert "Total Agents: 2" in conv_text
    assert "Analyst" in conv_text
    assert "Writer" in conv_text


def test_list_all_agents_with_conversation_and_prompt_flags():
    """list_all_agents properly respects add_collaboration_prompt when add_to_conversation=True."""
    w1 = DummyWorker("Worker1", "Worker 1 desc")

    conv_with_prompt = Conversation()
    list_all_agents(
        [w1],
        conversation=conv_with_prompt,
        add_to_conversation=True,
        add_collaboration_prompt=True,
    )
    assert "Fundamental Collaboration Principles" in conv_with_prompt.get_str()

    conv_without_prompt = Conversation()
    list_all_agents(
        [w1],
        conversation=conv_without_prompt,
        add_to_conversation=True,
        add_collaboration_prompt=False,
    )
    assert "Fundamental Collaboration Principles" not in conv_without_prompt.get_str()


def test_list_all_agents_default_does_not_inject_collaboration_principles():
    """list_all_agents by default produces only the agent roster."""
    w1 = DummyWorker("Worker1", "Worker 1 desc")
    result = list_all_agents([w1], add_to_conversation=False)
    assert "Fundamental Collaboration Principles" not in result
    assert "Worker1" in result


def test_list_all_agents_falls_back_to_system_prompt():
    """list_all_agents uses system_prompt when description is missing."""
    w = DummyWorker("NoDescWorker", description=None, system_prompt="Custom prompt text")
    result = list_all_agents([w], add_to_conversation=False)
    assert "NoDescWorker" in result
    assert "Custom prompt text" in result


def test_list_worker_agents_returns_clean_roster():
    """HierarchicalSwarm.list_worker_agents produces roster without collaboration principles."""
    w1 = DummyWorker("Worker1", "Desc1")
    swarm = HierarchicalSwarm(agents=[w1], print_on=False)
    roster = swarm.list_worker_agents()
    assert "Worker1" in roster
    assert "Fundamental Collaboration Principles" not in roster
