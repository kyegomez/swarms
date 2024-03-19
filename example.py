from swarms import Agent, Anthropic
from langchain.tools import tool


# Tool
@tool
def search_api(query: str, max_results: int = 10):
    """
    Search the web for the query and return the top `max_results` results.
    """
    return f"Search API: {query} -> {max_results} results"


## Initialize the workflow
agent = Agent(
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
