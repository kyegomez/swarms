"""
Context shape for :class:`~swarms.structs.debate_with_judge.DebateWithJudge`.

Covers #2029: every structure used to render the shared conversation as
``"role: content"`` prose and hand it to ``agent.run(task=<one big string>)``.
In this structure the flattening was per-speaker f-strings - the Con agent was
handed the Pro argument inside its prompt, the judge was handed both - so the
model could not tell a peer's argument from its own instruction.

The stub agents here record what each ``run()`` was actually given, so the
assertions are about the request that would go on the wire, not about the
prompt constants.
"""

from swarms.structs.agent import Agent
from swarms.structs.debate_with_judge import DebateWithJudge


def _record(agent, calls):
    """Replace an agent's run() with a stub that records the context it got."""
    name = agent.agent_name
    turns = [0]

    def _run(task=None, messages=None, **kwargs):
        turns[0] += 1
        answer = f"{name}-answer-{turns[0]}"
        calls.append(
            {
                "agent": name,
                "task": task,
                "messages": list(messages or []),
                "answer": answer,
                "system_prompt": agent.system_prompt,
            }
        )
        # agent_answer reads the answer back out of short_memory, not out of
        # the return value, so the stub has to write it there too.
        agent.short_memory.add(role=name, content=answer)
        return answer

    agent.run = _run
    return agent


def _debate(max_loops=2):
    """A DebateWithJudge whose three agents all record what they were sent."""
    calls = []
    agents = []
    for name in ("Pro-Debater", "Con-Debater", "Debate-Judge"):
        agent = Agent(
            agent_name=name,
            agent_description=f"Recording {name}",
            system_prompt=f"You are {name}.",
            model_name="gpt-4o",
            max_loops=1,
            persistent_memory=False,
            print_on=False,
            autosave=False,
        )
        agents.append(_record(agent, calls))

    debate = DebateWithJudge(
        pro_agent=agents[0],
        con_agent=agents[1],
        judge_agent=agents[2],
        max_loops=max_loops,
        verbose=False,
    )
    return debate, calls


def _for(calls, name):
    return [c for c in calls if c["agent"] == name]


def test_con_reads_the_pro_argument_as_a_turn_not_inside_its_task():
    """The Pro argument reaches the Con agent as a turn, not pasted in."""
    debate, calls = _debate(max_loops=1)
    debate.run("Motion: open models will win")

    pro = _for(calls, "Pro-Debater")[0]
    con = _for(calls, "Con-Debater")[0]

    assert pro["answer"] not in str(con["task"])
    assert f"Pro-Debater: {pro['answer']}" in "\n".join(
        m["content"] for m in con["messages"]
    )


def test_judge_reads_both_arguments_as_turns_not_inside_its_task():
    """Both arguments reach the judge as turns, not as prompt sections."""
    debate, calls = _debate(max_loops=1)
    debate.run("Motion: open models will win")

    pro = _for(calls, "Pro-Debater")[0]
    con = _for(calls, "Con-Debater")[0]
    judge = _for(calls, "Debate-Judge")[0]

    judge_turns = "\n".join(m["content"] for m in judge["messages"])
    for speaker in (pro, con):
        assert speaker["answer"] not in str(judge["task"])
        assert (
            f"{speaker['agent']}: {speaker['answer']}" in judge_turns
        )


def test_every_turn_is_a_typed_message():
    """Nothing reaches an agent as a role-prefixed blob in the task."""
    debate, calls = _debate(max_loops=2)
    debate.run("Motion: open models will win")

    for call in calls:
        assert isinstance(call["messages"], list)
        for message in call["messages"]:
            assert isinstance(message, dict)
            assert message["role"] in ("user", "assistant")
            assert isinstance(message["content"], str)

    # The opening Pro turn has nothing before it, so it carries no history.
    assert _for(calls, "Pro-Debater")[0]["messages"] == []


def test_an_agent_sees_its_own_prior_argument_as_assistant():
    """An agent's own output must not come back labelled as the user's."""
    debate, calls = _debate(max_loops=2)
    debate.run("Motion: open models will win")

    pro_calls = _for(calls, "Pro-Debater")
    assert len(pro_calls) == 2

    assistant = [
        m["content"]
        for m in pro_calls[1]["messages"]
        if m["role"] == "assistant"
    ]
    assert pro_calls[0]["answer"] in assistant


def test_the_topic_stays_the_motion_across_loops():
    """The judge's synthesis is a turn, so it is not also restated as the topic."""
    motion = "Motion: a very distinctive claim about tungsten"
    debate, calls = _debate(max_loops=2)
    debate.run(motion)

    judge_first = _for(calls, "Debate-Judge")[0]
    for call in calls:
        if call["agent"] == "Pro-Debater":
            assert motion in str(call["task"])
        # The second-loop prompts must not carry the synthesis text as well.
        assert judge_first["answer"] not in str(call["task"])


def test_role_framing_costs_no_llm_call_and_does_not_stack():
    """Intros go in the system prompt, not into three discarded run() calls."""
    debate, calls = _debate(max_loops=1)

    debate.run("Motion: open models will win")
    # one Pro, one Con, one Judge -- no discarded intro calls
    assert len(calls) == 3

    first_pro_prompt = _for(calls, "Pro-Debater")[0]["system_prompt"]
    assert "You are Pro-Debater." in first_pro_prompt
    assert "arguing in favor" in first_pro_prompt

    debate.run("Motion: a second, unrelated debate")
    second_pro_prompt = _for(calls, "Pro-Debater")[1]["system_prompt"]

    assert second_pro_prompt.count("You are Pro-Debater.") == 1
    assert "Motion: open models will win" not in second_pro_prompt
    assert "Motion: a second, unrelated debate" in second_pro_prompt
