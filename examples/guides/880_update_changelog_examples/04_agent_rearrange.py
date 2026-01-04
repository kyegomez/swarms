"""
Agent Rearrangement Example

This example demonstrates AgentRearrange for dynamic reordering and
restructuring of agents at runtime using flow patterns.
"""

from swarms import Agent, AgentRearrange

researcher = Agent(
    agent_name="researcher",
    system_prompt="You research topics thoroughly and gather information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer = Agent(
    agent_name="writer",
    system_prompt="You write clear and engaging content based on research.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

reviewer = Agent(
    agent_name="reviewer",
    system_prompt="You review content for quality and accuracy.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

flow = "researcher -> writer, reviewer"

rearrange_system = AgentRearrange(
    agents=[researcher, writer, reviewer],
    flow=flow,
    max_loops=1,
    team_awareness=True,
)

task = "Research and write a report on artificial intelligence trends"
result = rearrange_system.run(task)

print("Agent Rearrangement Result:")
print(result)
