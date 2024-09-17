from swarms import Agent
from swarm_models import Anthropic, tool

# Model
llm = Anthropic(
    temperature=0.1,
)

"""
How to create tools:

1. Define a function that takes the required arguments with documentation and type hints.
2. Add the `@tool` decorator to the function.
3. Add the function to the `tools` list in the `Agent` class.
"""


# Tools
# Browser tools
@tool
def browser(query: str):
    """
    Opens a web browser and searches for the given query on Google.

    Args:
        query (str): The search query.

    Returns:
        str: A message indicating that the search is being performed.
    """
    import webbrowser

    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"Searching for {query} in the browser."


# Agent
agent = Agent(
    agent_name="Devin",
    system_prompt=(
        "Autonomous agent that can interact with humans and other"
        " agents. Be Helpful and Kind. Use the tools provided to"
        " assist the user. Return all code in markdown format."
    ),
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
    tools=[browser],
)

# Run the agent
out = agent.run("what's the weather in Miami?")
print(out)
