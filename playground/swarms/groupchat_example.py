import subprocess


from swarms import (
    Agent,
    Anthropic,
    GroupChat,
    GroupChatManager,
    tool,
)

# Model
llm = Anthropic(
    temperature=0.1,
)


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
    Search the query in the browser with the `browser` tool.

    Args:
        query (str): The query to search in the browser.

    Returns:
        str: The search results.
    """
    import webbrowser

    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"Searching for {query} in the browser."


@tool
def create_file(file_path: str, content: str):
    """
    Create a file using the file editor tool.

    Args:
        file_path (str): The path to the file.
        content (str): The content to write to the file.

    Returns:
        str: The result of the file creation operation.
    """
    with open(file_path, "w") as file:
        file.write(content)
    return f"File {file_path} created successfully."


@tool
def file_editor(file_path: str, mode: str, content: str):
    """
    Edit a file using the file editor tool.

    Args:
        file_path (str): The path to the file.
        mode (str): The mode to open the file in.
        content (str): The content to write to the file.

    Returns:
        str: The result of the file editing operation.
    """
    with open(file_path, mode) as file:
        file.write(content)
    return f"File {file_path} edited successfully."


# Agent
agent = Agent(
    agent_name="Devin",
    system_prompt=(
        "Autonomous agent that can interact with humans and other"
        " agents. Be Helpful and Kind. Use the tools provided to"
        " assist the user. Return all code in markdown format."
    ),
    llm=llm,
    max_loops=1,
    autosave=False,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    tools=[terminal, browser, file_editor, create_file],
)

# Agent
agent_two = Agent(
    agent_name="Devin Worker 2",
    system_prompt=(
        "Autonomous agent that can interact with humans and other"
        " agents. Be Helpful and Kind. Use the tools provided to"
        " assist the user. Return all code in markdown format."
    ),
    llm=llm,
    max_loops=1,
    autosave=False,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    tools=[terminal, browser, file_editor, create_file],
)


# Initialize the group chat
group_chat = GroupChat(
    agents=[agent, agent_two],
    max_round=2,
    admin_name="Supreme Commander Kye",
    group_objective="Research everyone at Goldman Sachs",
)

# Initialize the group chat manager
manager = GroupChatManager(groupchat=group_chat, selector=agent)

# Run the group chat manager on a task
out = manager("Generate a 10,000 word blog on health and wellness.")
print(out)
