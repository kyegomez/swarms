from swarms import Agent, AdvisorSwarm


def my_code_tool(code: str) -> str:
    """Placeholder for a code execution tool."""
    return code


def my_search_tool(query: str) -> str:
    """Placeholder for a search tool."""
    return query


executor = Agent(
    agent_name="CodeExecutor",
    model_name="claude-sonnet-4-6",
    tools=[my_code_tool, my_search_tool],
)

advisor = Agent(
    agent_name="ArchitectAdvisor",
    model_name="claude-opus-4-6",
)

swarm = AdvisorSwarm(
    executor_agent=executor,
    advisor_agent=advisor,
    max_advisor_uses=5,
)
result = swarm.run("Refactor the auth module to use OAuth2")
print(result)
