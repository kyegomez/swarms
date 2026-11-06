from swarms.structs.advisor_swarm import AdvisorSwarm


class _NamedAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.received = []
        self.short_memory = []

    def reset(self):
        self.short_memory = []

    def run(self, task=None, img=None, imgs=None, messages=None):
        self.received.append(messages)
        self.short_memory.extend(messages or [])
        self.short_memory.append(
            {"role": self.agent_name, "content": task}
        )
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


def test_each_turn_and_run_starts_from_an_empty_agent_memory():
    executor = _NamedAgent("Coder")
    swarm = AdvisorSwarm(
        executor_agent=executor,
        advisor_agent=_NamedAgent("Strategist"),
        max_loops=2,
    )

    swarm.run("Write a merge function")
    merge = [
        row
        for row in executor.short_memory
        if row["content"] == "Write a merge function"
    ]
    assert len(merge) == 1, executor.short_memory

    swarm.run("Write a sort function")
    assert not [
        row
        for row in executor.short_memory
        if row["content"] == "Write a merge function"
    ], executor.short_memory
