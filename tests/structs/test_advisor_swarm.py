from swarms.structs.advisor_swarm import AdvisorSwarm


class _NamedAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.received = []

    def run(self, task=None, img=None, imgs=None, messages=None):
        self.received.append(messages)
        return f"{self.agent_name} turn {len(self.received)}"


def test_custom_named_agents_get_their_own_turns_back_as_assistant():
    executor = _NamedAgent("Coder")
    advisor = _NamedAgent("Strategist")

    AdvisorSwarm(
        executor_agent=executor,
        advisor_agent=advisor,
        max_loops=2,
    ).run("Write a merge function")

    assert {
        "role": "assistant",
        "content": "Coder turn 1",
    } in executor.received[1], executor.received[1]
    assert {
        "role": "assistant",
        "content": "Strategist turn 1",
    } in advisor.received[1], advisor.received[1]
