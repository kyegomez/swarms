from swarms import Agent
from swarm_models import Anthropic, tool


# Tool
@tool  # Wrap the function with the tool decorator
def search_api(query: str, max_results: int = 10):
    """
    Search the web for the query and return the top `max_results` results.
    """
    return f"Search API: {query} -> {max_results} results"


## Initialize the workflow
agent = Agent(
    agent_name="Youtube Transcript Generator",
    agent_description=(
        "Generate a transcript for a youtube video on what swarms"
        " are!"
    ),
    llm=Anthropic(),
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    tools=[search_api],
)

# Run the workflow on a task
agent(
    "Generate a transcript for a youtube video on what swarms are!"
    " Output a <DONE> token when done."
)
