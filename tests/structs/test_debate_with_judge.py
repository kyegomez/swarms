from swarms import Agent, DebateWithJudge


def test_preset_agents_fill_only_the_roles_not_passed():
    pro = Agent(agent_name="My-Pro", max_loops=1, print_on=False)
    con = Agent(agent_name="My-Con", max_loops=1, print_on=False)

    debate = DebateWithJudge(
        pro_agent=pro, con_agent=con, verbose=False
    )

    assert debate.pro_agent is pro
    assert debate.con_agent is con
    assert debate.judge_agent.agent_name == "Debate-Judge"
