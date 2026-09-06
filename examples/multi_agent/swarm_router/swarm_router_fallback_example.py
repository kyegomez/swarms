"""
SwarmRouter with fallback swarms.

``fallback_swarms`` is an ordered list of swarm types to try when the primary
``swarm_type`` raises during a run. Each fallback is built from the same agents
and configuration; the first one to complete wins. If every swarm in the chain
fails, ``SwarmRouterRunError`` is raised naming each attempt.

After a run, ``router.active_swarm_type`` says which swarm actually served it
and ``router.fallback_attempts`` lists the ones that failed first.

Run:
    export OPENAI_API_KEY=...
    python examples/multi_agent/swarm_router/swarm_router_fallback_example.py
"""

from swarms import Agent, SwarmRouter

agents = [
    Agent(
        agent_name="Researcher",
        system_prompt="You gather the facts relevant to the task.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
    Agent(
        agent_name="Analyst",
        system_prompt="You weigh the evidence and draw conclusions.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
    Agent(
        agent_name="Writer",
        system_prompt="You turn conclusions into a clear brief.",
        model_name="gpt-5.4",
        max_loops=1,
    ),
]

router = SwarmRouter(
    name="resilient-router",
    agents=agents,
    swarm_type="HierarchicalSwarm",
    # Tried in order, only if the swarm before it raises.
    fallback_swarms=["SequentialWorkflow", "ConcurrentWorkflow"],
    max_loops=1,
)

result = router.run(
    "Assess whether a mid-size SaaS company should adopt usage-based pricing."
)

print(result)
print(f"\nserved by: {router.active_swarm_type}")
for attempt in router.fallback_attempts:
    print(
        f"failed first: {attempt['swarm_type']} -> {attempt['error']}"
    )
