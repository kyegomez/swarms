"""A board stored in SQLite, loaded back later, and continued by another agent."""

import os

from swarms import Agent, MessageBoard

DB_PATH = "runs/board.db"
os.makedirs("runs", exist_ok=True)

# First session: a planner opens a thread and posts a plan in it.
board = MessageBoard(
    name="planning",
    description="Event plans and their budgets",
    db_path=DB_PATH,
)
planner = Agent(
    agent_name="Planner",
    system_prompt=(
        "You plan events. Open a message board thread for the event with "
        "create_thread, labelled status=open, with a short day-by-day plan "
        "as the opening post."
    ),
    model_name="gpt-5.4",
    max_loops=2,
    tools=board.agent_tools,
)
planner.run("Plan a two-day team offsite for 12 people near Austin.")
board_id = board.id
board.close()

# Later, even in a new process: the board's id brings the whole board back.
for saved in MessageBoard.list_boards(DB_PATH):
    print(f"{saved['id']} | {saved['name']} | {saved['description']}")

reopened = MessageBoard.load(DB_PATH, board_id)
budgeter = Agent(
    agent_name="Budgeter",
    system_prompt=(
        "You check plans on the message board against a budget. List the "
        "threads labelled status=open, read each plan, reply with a version "
        "that fits the budget, and label it status=done."
    ),
    model_name="gpt-5.4",
    max_loops=5,
    tools=reopened.agent_tools,
)
budgeter.run("Fit every open plan to a total budget of $6,000.")

for thread in reopened.threads():
    print(f"=== {thread['title']} {thread['labels']}")
    print(reopened.thread(thread["id"]).return_history_as_string())
print(f"Every board, thread and message is stored in {DB_PATH}")
