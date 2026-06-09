"""
Engineering design review: four engineers rotate through a proposed system
design and refine it across two passes.

The architect opens with the proposal, the SRE attacks operability, the
security engineer surfaces threat model gaps, and the staff engineer
adjudicates trade-offs. Loop 2 lets each role respond to the others'
critiques.
"""

from swarms import Agent, RoundRobinSwarm

architect = Agent(
    agent_name="Architect",
    agent_description="Proposes and defends the system design.",
    system_prompt=(
        "You are the proposing architect. State the design crisply: "
        "components, data flow, key APIs, and the failure modes you "
        "are willing to accept. In loop 2, revise based on the critique "
        "you received."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)

sre = Agent(
    agent_name="SRE",
    agent_description="Site reliability engineer.",
    system_prompt=(
        "You are an SRE. Stress-test the proposal on operability: SLOs, "
        "blast radius of failures, rollback story, observability gaps, "
        "and on-call burden. Give specific, actionable pushback."
    ),
    model_name="claude-sonnet-4-6",
    max_loops=1,
)

security = Agent(
    agent_name="SecurityEngineer",
    agent_description="Application security engineer.",
    system_prompt=(
        "You are a security engineer. Walk the threat model: auth, "
        "authz, data exposure, supply chain, and trust boundaries. "
        "Name at least one concrete attack the prior speakers missed."
    ),
    model_name="gemini/gemini-2.5-pro",
    max_loops=1,
)

staff_engineer = Agent(
    agent_name="StaffEngineer",
    agent_description="Senior IC arbitrating trade-offs.",
    system_prompt=(
        "You are a staff engineer. Adjudicate: which critiques are "
        "blocking, which are nice-to-have, and what is the smallest "
        "change to the design that addresses the blocking ones. End "
        "your turn with a clear ship / iterate / reject verdict."
    ),
    model_name="claude-opus-4-7",
    max_loops=1,
)

review = RoundRobinSwarm(
    name="System-Design-Review",
    agents=[architect, sre, security, staff_engineer],
    max_loops=2,
    output_type="all",
    verbose=True,
)

proposal = (
    "Proposal: replace our monolithic order-processing service with an "
    "event-driven architecture. Orders are published to a Kafka topic, "
    "consumed by three independent services (inventory, payments, "
    "fulfilment), each with their own Postgres database. A new GraphQL "
    "gateway aggregates state for the UI. Target: 10x throughput, "
    "independent deploys per service. We have 6 engineers and 8 weeks."
)

result = review.run(proposal)
print(result)
