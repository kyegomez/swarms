from swarms import Agent
from swarms_tools import exa_search


agent = Agent(
    name="Exa Search Agent",
    model_name="gpt-4o-mini",
    tools=[exa_search],
    tool_call_summary=False,
)

agent.run("What are the latest experimental treatments for diabetes?")
