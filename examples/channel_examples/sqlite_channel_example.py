from swarms import Agent, SequentialWorkflow, SqliteChannel

DB = "runs/channels.db"

researcher = Agent(
    agent_name="Researcher",
    system_prompt="Gather the key facts as a short bullet list.",
    model_name="gpt-5.4",
    max_loops=1,
)
analyst = Agent(
    agent_name="Analyst",
    system_prompt="Weigh the facts in the conversation and state the main risk.",
    model_name="gpt-5.4",
    max_loops=1,
)
writer = Agent(
    agent_name="Writer",
    system_prompt="Turn the conversation into a three-paragraph brief.",
    model_name="gpt-5.4",
    max_loops=1,
)

# One file, many conversations: this one is keyed by name.
channel = SqliteChannel(DB, name="ev-batteries")

workflow = SequentialWorkflow(
    agents=[researcher, analyst, writer],
    max_loops=1,
    channel=channel,
)
workflow.run("Report on the EV battery market.")

# The whole conversation, with ids and timestamps.
for row in channel.fetch_messages():
    print(
        f"\n[{row['created_at']}] {row['agent_name']}\n{row['content']}"
    )

# Every conversation stored in this file.
print(
    "Conversations in", DB, "->", SqliteChannel.list_conversations(DB)
)
