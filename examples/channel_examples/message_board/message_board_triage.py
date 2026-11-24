"""A bug queue that agents run on a message board, with labels as status."""

import os

from swarms import Agent, MessageBoard

os.makedirs("runs", exist_ok=True)
board = MessageBoard(
    name="bugs",
    description="Bug reports from the support inbox",
    db_path="runs/bugs.db",
)

support_inbox = """
1. Users on slow networks see a timeout after 5 seconds on /login.
2. Exported names like 'José' come out as 'Jos?' in the CSV export.
3. The page flashes white for a moment before dark mode loads.
"""


def board_agent(
    name: str, system_prompt: str, max_loops: int
) -> Agent:
    """Build an agent that works on the board through its own tools."""
    return Agent(
        agent_name=name,
        system_prompt=system_prompt,
        model_name="gpt-5.4",
        max_loops=max_loops,
        tools=board.agent_tools,
    )


# The triager opens one thread per bug.
triager = board_agent(
    "Triager",
    "You turn bug reports into message board threads. Open one thread "
    "per bug with create_thread, with a short title, the report as the "
    "opening post, and the labels kind=bug, status=open.",
    max_loops=2,
)
triager.run(
    f"Open a thread for each bug in this inbox:\n{support_inbox}"
)

# A fixer takes each open bug in turn.
for thread in board.threads({"kind": "bug", "status": "open"}):
    fixer = board_agent(
        "Fixer",
        "You are a senior engineer. Read the thread, reply with the likely "
        "cause and a concrete fix, then label the thread status=fixed.",
        max_loops=4,
    )
    fixer.run(f"Work on thread {thread['id']}.")

# A reviewer closes each fix it approves and reopens the rest.
reviewer = board_agent(
    "Reviewer",
    "You review fixes on the message board. List the threads labelled "
    "status=fixed, read each one, reply with your verdict, and label it "
    "status=closed if the fix is sound or status=open if it is not.",
    max_loops=8,
)
reviewer.run("Review every fixed thread on the board.")

for status in ("closed", "fixed", "open"):
    titles = [t["title"] for t in board.threads({"status": status})]
    print(f"{status}: {titles}")
