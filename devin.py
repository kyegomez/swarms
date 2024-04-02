"""
Plan -> act in a loop until observation is met


# Tools
- Terminal
- Text Editor
- Browser
"""
from swarms import Agent, OpenAIChat, tool
import subprocess

# Model
llm = OpenAIChat()


# Tools
@tool
def terminal(
    code: str,
):
    """
    Run code in the terminal.

    Args:
        code (str): The code to run in the terminal.

    Returns:
        str: The output of the code.
    """
    out = subprocess.run(
        code, shell=True, capture_output=True, text=True
    ).stdout
    return str(out)


@tool
def browser(query: str):
    """
    Search the query in the browser.

    Args:
        query (str): The query to search in the browser.

    Returns:
        str: The search results.
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
        " assist the user."
    ),
    llm=llm,
    max_loops=4,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
    tools=[terminal, browser],
    # streaming=True,
)

# Run the agent
out = agent("What is the weather today in palo alto?")
print(out)
