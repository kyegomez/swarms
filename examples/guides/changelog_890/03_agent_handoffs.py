from swarms import Agent

# Create specialized agents for different tasks
research_agent = Agent(
    agent_name="ResearchAgent",
    system_prompt="You are a research specialist. Provide comprehensive, well-researched information on any topic.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writing_agent = Agent(
    agent_name="WritingAgent",
    system_prompt="You are a professional writer. Create clear, engaging content based on provided information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create the main coordinator agent with handoff capabilities
coordinator_agent = Agent(
    agent_name="ProjectCoordinator",
    system_prompt="""You are a project coordinator that delegates tasks to specialized agents.
    Use the handoff_task tool to delegate work to ResearchAgent and WritingAgent based on their expertise.
    Always provide clear reasoning for each handoff.""",
    model_name="gpt-4o-mini",
    max_loops=1,
    handoffs=[research_agent, writing_agent],
)

# Complex task requiring multiple specialized skills
project_task = """
Call the writing agent and ask it to write a report about the best performing oil stocks in Venezuela.
"""

response = coordinator_agent.run(project_task)
print(response)