"""
Agent Rearrange Patterns Example

This example demonstrates dynamic agent reconfiguration and task reallocation
patterns using AgentRearrange. Shows sequential and concurrent execution
with custom flow patterns.
"""

from swarms import Agent, AgentRearrange

# Create specialized agents for different tasks
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist. Gather comprehensive information on given topics.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You are a professional writer. Create clear, engaging content from research.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

reviewer = Agent(
    agent_name="Reviewer",
    system_prompt="You are a content reviewer. Provide feedback and improvement suggestions.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create agent rearrange system with flow pattern
# Flow: Researcher -> Writer and Reviewer (concurrent) -> Final synthesis
rearrange_system = AgentRearrange(
    name="ContentCreationRearrange",
    description="Dynamic agent reconfiguration for content creation workflow",
    agents=[researcher, writer, reviewer],
    flow="Researcher -> Writer, Reviewer",  # Sequential then concurrent execution
    max_loops=1,
    team_awareness=True,
)

task = "Research renewable energy trends, write a summary article, and provide editorial feedback."
result = rearrange_system.run(task)
print(result)