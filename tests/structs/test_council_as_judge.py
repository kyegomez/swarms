from swarms.structs.council_as_judge import CouncilAsAJudge


def _stub_run(agent, reply, calls):
    def _run(task=None, messages=None, **kwargs):
        calls.append(
            {
                "agent": agent.agent_name,
                "task": task,
                "messages": list(messages or []),
            }
        )
        return reply

    agent.run = _run


def _build_council():
    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        random_model_name=False,
        aggregation_model_name="gpt-4o-mini",
        judge_agent_model_name="gpt-4o-mini",
    )
    calls = []
    for agent in council.judge_agents.values():
        _stub_run(agent, f"{agent.agent_name}-rationale", calls)
    _stub_run(council.aggregator_agent, "final report", calls)
    return council, calls


def test_aggregator_receives_typed_turns_per_dimension():
    council, calls = _build_council()

    council.run(task="Evaluate this response for quality.")

    aggregator_calls = [
        c
        for c in calls
        if c["agent"] == council.aggregator_agent.agent_name
    ]
    assert len(aggregator_calls) == 1
    call = aggregator_calls[0]
    turns = call["messages"] + [
        {"role": "user", "content": str(call["task"])}
    ]

    for message in turns:
        assert isinstance(message, dict)
        assert message["role"] in ("user", "assistant", "system")
        assert isinstance(message["content"], str)

    contents = [m["content"] for m in turns]
    for agent in council.judge_agents.values():
        assert any(
            f"{agent.agent_name}: {agent.agent_name}-rationale"
            in text
            for text in contents
        ), f"no turn attributed to {agent.agent_name}: {contents}"

    assert len(turns) == len(council.judge_agents) + 1
