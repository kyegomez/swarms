from swarms import Agent, DecisionModel

agents = [
    Agent(
        agent_name="Billing-Agent",
        agent_description="Refunds, invoices, failed payments and subscription changes.",
        system_prompt="You resolve billing questions clearly and briefly.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
    Agent(
        agent_name="Technical-Agent",
        agent_description="Bugs, API errors, integrations and outages.",
        system_prompt="You debug technical problems step by step.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
    Agent(
        agent_name="Sales-Agent",
        agent_description="Pricing, plan upgrades and new accounts.",
        system_prompt="You answer pricing and plan questions.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
]
agents_by_name = {agent.agent_name: agent for agent in agents}

# Any name from get_decision_models() works, e.g. "clef" for Cloudflare's Clef.
router = DecisionModel(model_name="jev-latest")


def route(task: str) -> str:
    """
    Send a task to the best agent, or escalate when the router is unsure.

    Args:
        task: The customer request.

    Returns:
        The agent's reply or an escalation notice.
    """
    answers = router.run(
        state=task,
        questions={
            "agent": {
                "type": "choice",
                "instructions": "Which agent should handle this request?",
                "criteria": {
                    agent.agent_name: agent.agent_description
                    for agent in agents
                },
            },
            "in_scope": {
                "type": "noul",
                "instructions": "This is a request a software company's support team should handle.",
            },
        },
    )["answers"]

    if answers["in_scope"]["noul"] < 0.5:
        return "Declined: out of scope for support."

    decision = answers["agent"]
    if decision["confidence"] < 0.5:
        return (
            f"Escalated to a human: confidence {decision['confidence']:.2f} "
            f"across {decision['probabilities']}."
        )

    print(
        f"Routing to {decision['choice']} "
        f"(confidence {decision['confidence']:.2f})"
    )
    return agents_by_name[decision["choice"]].run(task)


tasks = [
    "I was charged twice for my March invoice, can you refund one?",
    "Our webhook endpoint returns 500 since your API update this morning.",
    "What does the enterprise plan cost for 200 seats?",
    "Can you write my history essay on the French Revolution?",
]

for task in tasks:
    print(f"\nTask: {task}")
    print(route(task))
