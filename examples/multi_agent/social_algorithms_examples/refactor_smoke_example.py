"""Minimal example for `SocialAlgorithms`.

Defines a tiny custom algorithm where a researcher hands off to an analyst
who hands off to a writer. Demonstrates adding and removing agents and
reading their names from the swarm's agent list.
"""

from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


def research_analysis_writeup(agents, task, **_kwargs):
    research = agents[0].run(f"Research: {task}")
    analysis = agents[1].run(f"Analyze this research:\n{research}")
    writeup = agents[2].run(
        f"Write a one-paragraph summary based on:\n{analysis}"
    )
    return {
        "research": research,
        "analysis": analysis,
        "writeup": writeup,
    }


def main() -> None:
    researcher = Agent(
        agent_name="Researcher",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )
    analyst = Agent(
        agent_name="Analyst",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )
    writer = Agent(
        agent_name="Writer",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )

    swarm = SocialAlgorithms(
        name="ResearchAnalysisWriteup",
        agents=[researcher, analyst, writer],
        social_algorithm=research_analysis_writeup,
        verbose=False,
    )

    print(
        "agent names:", [agent.agent_name for agent in swarm.agents]
    )

    extra = Agent(
        agent_name="Reviewer",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )
    swarm.add_agent(extra)
    print("after add:", [agent.agent_name for agent in swarm.agents])
    swarm.remove_agent("Reviewer")
    print(
        "after remove:", [agent.agent_name for agent in swarm.agents]
    )

    result = swarm.run("the rise of vector databases")
    print("final writeup:\n", result.final_outputs["writeup"])


if __name__ == "__main__":
    main()
