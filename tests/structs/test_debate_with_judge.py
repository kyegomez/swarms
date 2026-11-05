from swarms import Agent, DebateWithJudge


def _agent(name, reply):
    agent = Agent(
        agent_name=name,
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )
    agent.call_llm = lambda *args, **kwargs: reply
    return agent


def test_default_output_keeps_the_first_pro_argument():
    debate = DebateWithJudge(
        agents=[
            _agent("Pro", "pro argument"),
            _agent("Con", "con argument"),
            _agent("Judge", "judge synthesis"),
        ],
        max_loops=1,
        verbose=False,
    )

    result = debate.run("Is remote work better than office work?")

    assert result == "pro argument\ncon argument\njudge synthesis"
