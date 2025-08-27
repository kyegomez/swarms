from swarms import Agent
from swarms_tools import exa_search


agent = Agent(
    name="Exa Search Agent",
    llm="gpt-4o-mini",
    tools=[exa_search],
)

agent.run("What are the latest experimental treatments for diabetes?")