from typing import List, Dict, Any
from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


def peer_review_algorithm(
    agents: List[Agent], task: str, **kwargs
) -> Dict[str, Any]:
    """
    A peer review social algorithm where agents review each other's work.

    Args:
        agents: List of agents participating in the algorithm
        task: The task to be processed
        **kwargs: Additional keyword arguments

    Returns:
        Dict containing the results from each agent and their reviews
    """
    if len(agents) < 2:
        raise ValueError("This algorithm requires at least 2 agents")

    results = {}
    reviews = {}

    # Each agent works on the task independently
    for i, agent in enumerate(agents):
        agent_prompt = f"Work on the following task: {task}"
        result = agent.run(agent_prompt)
        results[f"agent_{i}_{agent.agent_name}"] = result

    # Each agent reviews another agent's work (circular review)
    for i, agent in enumerate(agents):
        reviewer_index = (i + 1) % len(agents)
        reviewed_agent = agents[reviewer_index]

        review_prompt = f"Review the following work by {reviewed_agent.agent_name}:\n\n{results[f'agent_{reviewer_index}_{reviewed_agent.agent_name}']}\n\nProvide constructive feedback and suggestions for improvement."
        review = agent.run(review_prompt)
        reviews[
            f"{agent.agent_name}_reviews_{reviewed_agent.agent_name}"
        ] = review

    return {
        "original_work": results,
        "peer_reviews": reviews,
        "task": task,
    }


# Create agents
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist focused on gathering comprehensive information.",
    model_name="anthropic/claude-haiku-4-5-20251001",
    max_loops=1,
    top_p=None,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are an analytical specialist focused on interpreting and analyzing data.",
    model_name="anthropic/claude-haiku-4-5-20251001",
    max_loops=1,
    top_p=None,
)

reviewer = Agent(
    agent_name="Reviewer",
    system_prompt="You are a quality reviewer focused on providing constructive feedback.",
    model_name="anthropic/claude-haiku-4-5-20251001",
    max_loops=1,
    top_p=None,
)

# Create and run the social algorithm
social_alg = SocialAlgorithms(
    name="Peer-Review",
    description="Peer review workflow where agents review each other's work",
    agents=[researcher, analyst, reviewer],
    social_algorithm=peer_review_algorithm,
    verbose=False,
)

result = social_alg.run("What are the best gold ETFS")

print(result)
