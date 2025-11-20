from typing import List, Dict, Any
from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


def collaborative_brainstorming_algorithm(
    agents: List[Agent], task: str, **kwargs
) -> Dict[str, Any]:
    """
    A collaborative brainstorming algorithm where agents build upon each other's ideas.

    Args:
        agents: List of agents participating in the algorithm
        task: The task to be processed
        **kwargs: Additional keyword arguments

    Returns:
        Dict containing the brainstorming results
    """
    if len(agents) < 2:
        raise ValueError("This algorithm requires at least 2 agents")

    ideas = []
    current_context = task

    # Each agent contributes ideas building on previous ones
    for i, agent in enumerate(agents):
        if i == 0:
            # First agent starts with initial ideas
            prompt = (
                f"Brainstorm initial ideas for: {current_context}"
            )
        else:
            # Subsequent agents build on previous ideas
            previous_ideas = "\n".join(
                [f"- {idea}" for idea in ideas]
            )
            prompt = f"Building on these previous ideas:\n{previous_ideas}\n\nGenerate additional creative ideas and improvements for: {current_context}"

        agent_ideas = agent.run(prompt)
        ideas.append(
            f"Agent {i+1} ({agent.agent_name}): {agent_ideas}"
        )

        # Update context for next agent
        current_context = (
            f"{current_context}\n\nPrevious ideas: {agent_ideas}"
        )

    # Final synthesis by the last agent
    final_agent = agents[-1]
    all_ideas = "\n".join(ideas)
    synthesis_prompt = f"Review all the brainstorming ideas and create a final comprehensive solution:\n\n{all_ideas}"
    final_synthesis = final_agent.run(synthesis_prompt)

    return {
        "individual_ideas": ideas,
        "final_synthesis": final_synthesis,
        "task": task,
    }


# Create agents
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist focused on gathering comprehensive information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are an analytical specialist focused on interpreting and analyzing data.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

synthesizer = Agent(
    agent_name="Synthesizer",
    system_prompt="You are a synthesis specialist focused on combining information into coherent outputs.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create and run the social algorithm
social_alg = SocialAlgorithms(
    name="Collaborative-Brainstorming",
    description="Collaborative brainstorming where agents build on each other's ideas",
    agents=[researcher, analyst, synthesizer],
    social_algorithm=collaborative_brainstorming_algorithm,
    verbose=True,
)

result = social_alg.run("Innovative solutions for climate change")
