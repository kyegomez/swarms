"""A fan-out posts from several threads into one SQLite table, then a query reads it back."""

import sqlite3

from swarms import Agent, ConcurrentWorkflow, SqliteChannel

angles = ["technology", "regulation", "supply chain"]
agents = [
    Agent(
        agent_name=f"{angle.title()}-Analyst",
        system_prompt=f"Analyse the task from the {angle} angle only.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    for angle in angles
]

# One file can hold many conversations; this one is named after the quarter.
channel = SqliteChannel("runs/channels.db", name="q3-ev-batteries")

workflow = ConcurrentWorkflow(agents=agents, channel=channel)
workflow.run(
    "What will shape the EV battery market over the next five years?"
)

for message in channel.messages():
    print(f"\n[{message['agent_name']}]\n{message['content']}")

# The table is ordinary SQLite, so any client can query it.
rows = sqlite3.connect("runs/channels.db").execute(
    "SELECT agent_name, created_at FROM messages WHERE channel = ? ORDER BY id",
    ("q3-ev-batteries",),
)
for agent_name, created_at in rows:
    print(f"{created_at}  {agent_name}")
