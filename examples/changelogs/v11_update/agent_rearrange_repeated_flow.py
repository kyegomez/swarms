from swarms import Agent, AgentRearrange

agent_a = Agent(agent_name="AgentA", model_name="gpt-5.4")
agent_b = Agent(agent_name="AgentB", model_name="claude-sonnet-4-6")
agent_c = Agent(agent_name="AgentC", model_name="claude-sonnet-4-6")

# Flow with repeated agent — now works correctly
pipeline = AgentRearrange(
    agents=[agent_a, agent_b, agent_c],
    flow="AgentA -> AgentB -> AgentA -> AgentC",
)
result = pipeline.run("task")
# AgentA's second appearance correctly receives AgentB's output, not the original task
print(result)
