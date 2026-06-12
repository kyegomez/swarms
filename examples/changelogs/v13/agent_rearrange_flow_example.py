"""AgentRearrange flow DSL upgrades (v12.0.1).

- Pure concurrent flows: a flow may consist solely of comma-separated
  agents — no "->" operator required.
- explain() prints the parsed execution plan.
- output_type is validated against the allowed literal set immediately,
  and set_custom_flow validates the flow on update instead of at run time.
"""

from swarms import Agent, AgentRearrange

a = Agent(agent_name="A", model_name="gpt-5.4", max_loops=1)
b = Agent(agent_name="B", model_name="gpt-5.4", max_loops=1)
c = Agent(agent_name="C", model_name="gpt-5.4", max_loops=1)

# All three run in parallel — no "->" needed
flow = AgentRearrange(agents=[a, b, c], flow="A, B, C")

# Print the execution plan before running anything
flow.explain()

result = flow.run("Brainstorm product names.")
print(result)

# output_type is checked against the allowed literal set up front
try:
    AgentRearrange(agents=[a, b], flow="A -> B", output_type="bogus")
except ValueError as error:
    print(f"Invalid output_type rejected immediately: {error}")
