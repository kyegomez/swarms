from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange
from swarms.structs.sequential_workflow import SequentialWorkflow


def _make_recording_agents(names):
    calls = []
    agents = []
    for name in names:
        agent = Agent(
            agent_name=name,
            system_prompt=f"You are {name}.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=False,
        )

        def _make(agent_obj, agent_name):
            def _run(task=None, messages=None, **kwargs):
                calls.append(
                    {
                        "agent": agent_name,
                        "task": task,
                        "messages": list(messages or []),
                    }
                )
                answer = f"{agent_name}-answer"
                agent_obj.short_memory.add(role=agent_name, content=answer)
                return answer

            return _run

        agent.run = _make(agent, name)
        agents.append(agent)
    return agents, calls


def test_team_awareness_true_delivers_flow_structure_to_all_agents():
    """When team_awareness is True, every agent receives the complete Sequential Flow Structure."""
    agents, calls = _make_recording_agents(["A", "B", "C", "D"])
    rearrange = AgentRearrange(
        agents=agents,
        flow="A -> B, C -> D",
        max_loops=1,
        team_awareness=True,
    )
    rearrange.run("Process order")

    assert len(calls) == 4
    for call in calls:
        system_turns = [
            m["content"] for m in call["messages"] if m["role"] == "system"
        ]
        assert len(system_turns) >= 1
        combined_system = "\n\n".join(system_turns)
        assert "Sequential Flow Structure:" in combined_system
        assert "Step 1: A (leads to: B, C)" in combined_system
        assert "Step 2: B, C (follows: A) (leads to: D)" in combined_system
        assert "Step 3: D (follows: B, C)" in combined_system


def test_team_awareness_false_does_not_deliver_flow_structure():
    """When team_awareness is False, the full pipeline flow structure is not delivered."""
    agents, calls = _make_recording_agents(["A", "B"])
    rearrange = AgentRearrange(
        agents=agents,
        flow="A -> B",
        max_loops=1,
        team_awareness=False,
    )
    rearrange.run("Process order")

    assert len(calls) == 2
    for call in calls:
        system_turns = [
            m["content"] for m in call["messages"] if m["role"] == "system"
        ]
        combined_system = "\n\n".join(system_turns)
        assert "Sequential Flow Structure:" not in combined_system


def test_sequential_workflow_team_awareness_delivers_flow_structure():
    """SequentialWorkflow with team_awareness=True delivers flow structure."""
    agents, calls = _make_recording_agents(["Writer", "Reviewer"])
    workflow = SequentialWorkflow(
        agents=agents,
        max_loops=1,
        team_awareness=True,
    )
    workflow.run("Write essay")

    assert len(calls) == 2
    for call in calls:
        system_turns = [
            m["content"] for m in call["messages"] if m["role"] == "system"
        ]
        combined_system = "\n\n".join(system_turns)
        assert "Sequential Flow Structure:" in combined_system
        assert "Step 1: Writer (leads to: Reviewer)" in combined_system
        assert "Step 2: Reviewer (follows: Writer)" in combined_system
