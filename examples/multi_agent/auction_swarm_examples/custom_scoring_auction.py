from swarms.structs.agent import Agent
from swarms.structs.auction_swarm import AuctionSwarm


def quality_first(confidence: float, estimated_cost: float) -> float:
    """Score bids by quality first, cost second.

    Confidence dominates the score (squared), and cost only applies a gentle
    penalty. Use this when you care more about getting the best answer than
    about saving tokens.

    Args:
        confidence: Agent's self-assessed confidence, 0..1.
        estimated_cost: Agent's self-assessed relative cost (1.0 = average).

    Returns:
        A score used to rank bids, highest wins.
    """
    return (confidence**2) / (estimated_cost**0.25)


# A pool of writers with different strengths. For a technical explainer,
# more than one may be a reasonable fit — top_k=2 lets the two best both
# attempt it, and the auctioneer keeps the best result.
technical_writer = Agent(
    agent_name="Technical-Writer",
    agent_description="Precise technical documentation",
    system_prompt=(
        "You write precise, well-structured technical documentation for "
        "engineers. You value accuracy and clarity."
    ),
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

educator = Agent(
    agent_name="Educator",
    agent_description="Explains complex topics to beginners",
    system_prompt=(
        "You are a teacher who explains complex topics to beginners using "
        "simple language and analogies."
    ),
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

marketer = Agent(
    agent_name="Marketer",
    agent_description="Persuasive marketing copy",
    system_prompt=(
        "You write punchy, persuasive marketing copy designed to sell."
    ),
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

swarm = AuctionSwarm(
    name="quality-first-auction",
    agents=[technical_writer, educator, marketer],
    top_k=2,  # the two best bidders both run; best result is kept
    scoring=quality_first,  # custom ranking function
    print_on=True,
)

if __name__ == "__main__":
    result = swarm.run(
        "Explain how a hash map works to a first-year computer science "
        "student, in about one paragraph."
    )
    print(result)
