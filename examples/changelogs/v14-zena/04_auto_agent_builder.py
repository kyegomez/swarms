"""Zena: AutoAgentBuilder — generate the roster, keep the architecture.

A builder agent is forced to call one function, build_agents, so the
provider enforces the schema. Each generated agent carries exactly the four
fields Agent needs.

Note max_agents vs num_agents: max_agents is a CEILING and the builder
prefers the smallest roster that covers the task, so max_agents=5 on a
three-role problem returns three. num_agents is a hard requirement.
"""

from swarms import AutoAgentBuilder, SequentialWorkflow

TASK = (
    "Analyze why a B2B SaaS company's churn increased last quarter, "
    "and write a short brief for the leadership team."
)

# --- Inspect the roster: return_dict=True constructs nothing ----------
for config in AutoAgentBuilder(max_agents=3, return_dict=True).run(
    TASK
):
    print(f"{config['name']}  [{config['model_name']}]")
    print(f"  {config['description']}\n")

# --- Ceiling vs exact count ------------------------------------------
ceiling = AutoAgentBuilder(max_agents=5, return_dict=True).run(TASK)
exact = AutoAgentBuilder(num_agents=5, return_dict=True).run(TASK)
print(f"max_agents=5 -> {len(ceiling)} agents")
print(f"num_agents=5 -> {len(exact)} agents")

# --- Live Agent objects, run in sequence ------------------------------
agents = AutoAgentBuilder(
    num_agents=3, agent_kwargs={"max_loops": 1}
).run(TASK)
print(SequentialWorkflow(agents=agents, max_loops=1).run(TASK))
