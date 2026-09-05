"""
router.usage — token usage for the whole swarm.

``SwarmRouter.usage`` sums ``Agent.usage`` over every agent its swarms have
run: the agents you passed in, plus any the swarm built for itself — a
HierarchicalSwarm director, a MixtureOfAgents aggregator, a judge. Same
four keys as the agent version:

    {"input_tokens": ..., "output_tokens": ..., "cached_tokens": ..., "total_tokens": ...}

Agent totals are lifetime totals, so an agent shared between two routers
contributes what it spent in both. Streaming calls are not counted.

Run:
    export OPENAI_API_KEY=...
    python examples/multi_agent/swarm_router/swarm_router_usage.py
"""

from swarms import Agent, SwarmRouter

# $ per 1M tokens — substitute your provider's rates.
INPUT_PRICE, OUTPUT_PRICE = 2.50, 10.00


def cost(usage: dict) -> float:
    return (
        usage["input_tokens"] * INPUT_PRICE
        + usage["output_tokens"] * OUTPUT_PRICE
    ) / 1_000_000


def show(label: str, usage: dict) -> None:
    print(
        f"  {label:<22} in={usage['input_tokens']:>6}  "
        f"out={usage['output_tokens']:>6}  "
        f"total={usage['total_tokens']:>6}  ${cost(usage):.5f}"
    )


def make_agents():
    return [
        Agent(
            agent_name="Researcher",
            system_prompt="Gather the key facts. Be brief.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
        Agent(
            agent_name="Analyst",
            system_prompt="Weigh the facts you were given. Be brief.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
        Agent(
            agent_name="Writer",
            system_prompt="Write a three-sentence brief from the analysis.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
    ]


TASK = "Should a small team adopt a monorepo? Give a recommendation."


# --- 1. Sequential: the total is the sum of the three agents ---------------

agents = make_agents()
router = SwarmRouter(
    name="usage-sequential",
    agents=agents,
    swarm_type="SequentialWorkflow",
    max_loops=1,
)

print("SequentialWorkflow")
show("before run", router.usage)
router.run(TASK)

for agent in agents:
    show(agent.agent_name, agent.usage)
show("router.usage", router.usage)


# --- 2. Hierarchical: the director is counted too --------------------------
#
# The director is built by the swarm, not passed in, so it is not in
# router.agents. router.usage still includes it.

hierarchical = SwarmRouter(
    name="usage-hierarchical",
    agents=make_agents(),
    swarm_type="HierarchicalSwarm",
    max_loops=1,
)

print("\nHierarchicalSwarm")
hierarchical.run(TASK)

for agent in hierarchical.agents:
    show(agent.agent_name, agent.usage)
show("router.usage", hierarchical.usage)

workers = sum(a.usage["total_tokens"] for a in hierarchical.agents)
director = hierarchical.usage["total_tokens"] - workers
print(
    f"  director's share: {director} tokens, not in any worker's usage"
)


# --- 3. The cost of one run on a router that has run before ----------------

before = router.usage
router.run("Same question, but for a team of fifty.")
after = router.usage

print("\nSecond sequential run")
show("this run only", {k: after[k] - before[k] for k in after})
show("lifetime", after)
