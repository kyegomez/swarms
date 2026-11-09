"""Two agents talk through a markdown file instead of in-memory conversation."""

from swarms import Agent, FileChannel

# Every agent given this channel reads it before a run and posts its answer after.
channel = FileChannel("runs/ev_conversation.md")

researcher = Agent(
    agent_name="Researcher",
    system_prompt="You gather facts and figures. Answer in a short bullet list.",
    model_name="claude-sonnet-5",
    max_loops=1,
    persistent_memory=False,
    channel=channel,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You turn the facts in the conversation into a short report.",
    model_name="claude-sonnet-5",
    max_loops=1,
    persistent_memory=False,
    channel=channel,
)

task = "Let's have a conversation about electric vehicles (EVs) and their impact. Discuss the current trends, challenges, and future prospects of EV technology."

for _ in range(2):  # Each agent gets a few turns to 'chat'
    researcher.run(task)
    writer.run(task)

print(channel.path.read_text())
