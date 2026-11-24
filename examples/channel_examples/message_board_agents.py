"""A planner agent opens threads on a board; a researcher answers them."""

from swarms import Agent, MessageBoard

board = MessageBoard(
    name="research",
    description="One thread per question in a research brief",
)

# Each agent gets its own tool set, so every post it makes is signed with its name.
planner = Agent(
    agent_name="Planner",
    system_prompt=(
        "You split a research brief into separate questions. Open one "
        "message board thread per question with create_thread, labelled "
        "status=open."
    ),
    model_name="gpt-5.4",
    max_loops=2,
    tools=board.agent_tools,
)
researcher = Agent(
    agent_name="Researcher",
    system_prompt=(
        "You answer questions on the message board. List the threads "
        "labelled status=open, read each one, reply with a short answer, "
        "then label it status=answered."
    ),
    model_name="gpt-5.4",
    max_loops=6,
    tools=board.agent_tools,
)

planner.run(
    "Brief: how did EV battery prices change over the last two years, "
    "and which companies are adding the most AI chip capacity?"
)
researcher.run("Answer every open question on the board.")

for thread in board.threads():
    print(f"[{thread['id']}] {thread['title']} {thread['labels']}")
    print(board.thread(thread["id"]).return_history_as_string())
    print()
