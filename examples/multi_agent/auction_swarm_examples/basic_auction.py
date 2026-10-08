from swarms.structs.agent import Agent
from swarms.structs.auction_swarm import AuctionSwarm

# A pool of specialists with different areas of expertise. On a legal task the
# legal expert should bid high confidence; the others should bid low.
legal_expert = Agent(
    agent_name="Legal-Expert",
    agent_description="Contract law and plain-English legal explanations",
    system_prompt=(
        "You are a contract lawyer. You explain dense legal language in "
        "clear, plain English without losing important nuance."
    ),
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

finance_expert = Agent(
    agent_name="Finance-Expert",
    agent_description="Financial modeling and valuation",
    system_prompt=(
        "You are a financial analyst focused on valuation, modeling, and "
        "quantitative analysis."
    ),
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

generalist = Agent(
    agent_name="Generalist",
    agent_description="Broad general-purpose assistant",
    system_prompt="You are a helpful general-purpose assistant.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

# top_k=1 -> only the single best bidder runs the task.
swarm = AuctionSwarm(
    name="specialist-auction",
    agents=[legal_expert, finance_expert, generalist],
    top_k=1,
    print_on=True,  # print the bid table and the award
)

if __name__ == "__main__":
    result = swarm.run(
        "Translate the following clause into plain English: "
        "'The party of the first part shall indemnify and hold harmless "
        "the party of the second part from any and all liabilities arising "
        "hereunder.'"
    )
    print(result)
