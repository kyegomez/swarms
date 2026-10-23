"""A flow with a parallel step, recorded by the structure rather than the agents."""

from swarms import Agent, AgentRearrange, FileChannel

planner = Agent(
    agent_name="Planner", model_name="gpt-5.4", max_loops=1
)
coder = Agent(agent_name="Coder", model_name="gpt-5.4", max_loops=1)
reviewer = Agent(
    agent_name="Reviewer", model_name="gpt-5.4", max_loops=1
)
tester = Agent(agent_name="Tester", model_name="gpt-5.4", max_loops=1)

channel = FileChannel("runs/rearrange.md")

flow = AgentRearrange(
    agents=[planner, coder, reviewer, tester],
    flow="Planner -> Coder -> Reviewer, Tester",
    max_loops=1,
    channel=channel,
)
flow.run("Write a Python function that validates email addresses.")

print(channel.path.read_text())
