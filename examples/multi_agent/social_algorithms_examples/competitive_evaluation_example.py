from typing import List, Dict, Any
from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


def competitive_evaluation_algorithm(
    agents: List[Agent], task: str, **kwargs
) -> Dict[str, Any]:
    """
    A competitive evaluation algorithm where agents compete and are evaluated.

    Args:
        agents: List of agents participating in the algorithm
        task: The task to be processed
        **kwargs: Additional keyword arguments

    Returns:
        Dict containing the competitive results
    """
    if len(agents) < 3:
        raise ValueError(
            "This algorithm requires at least 3 agents (2 competitors + 1 judge)"
        )

    competitors = agents[:-1]
    judge = agents[-1]

    # Each competitor works on the task
    competitor_results = {}
    for i, competitor in enumerate(competitors):
        competitor_prompt = (
            f"Solve this task as best as you can: {task}"
        )
        result = competitor.run(competitor_prompt)
        competitor_results[
            f"competitor_{i+1}_{competitor.agent_name}"
        ] = result

    # Judge evaluates all solutions
    evaluation_prompt = f"Evaluate these solutions and rank them:\n\n"
    for name, result in competitor_results.items():
        evaluation_prompt += f"{name}:\n{result}\n\n"

    evaluation_prompt += "Provide rankings, scores, and detailed feedback for each solution."
    evaluation = judge.run(evaluation_prompt)

    return {
        "competitor_solutions": competitor_results,
        "judge_evaluation": evaluation,
        "task": task,
    }


# Create agents
competitor1 = Agent(
    agent_name="Competitor1",
    system_prompt="You are a competitive problem-solver focused on finding the best solution.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

competitor2 = Agent(
    agent_name="Competitor2",
    system_prompt="You are a competitive problem-solver focused on finding the best solution.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

judge = Agent(
    agent_name="Judge",
    system_prompt="You are an impartial judge focused on evaluating and ranking solutions objectively.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create and run the social algorithm
social_alg = SocialAlgorithms(
    name="Competitive-Evaluation",
    description="Competitive evaluation where agents compete and are judged",
    agents=[competitor1, competitor2, judge],
    social_algorithm=competitive_evaluation_algorithm,
    verbose=True,
)

result = social_alg.run(
    "Design the most efficient algorithm for sorting large datasets"
)