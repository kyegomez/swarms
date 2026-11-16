"""Two agents share an AgentMail email thread; the people in `to` can reply.

Set AGENTMAIL_API_KEY and create an inbox at https://agentmail.to first.
"""

from swarms import Agent, AgentMailChannel

channel = AgentMailChannel(
    inbox_id="research@agentmail.to",
    to="you@example.com",
    subject="EV battery market report",
)

researcher = Agent(
    agent_name="Researcher",
    system_prompt="You gather facts and figures. Answer in a short bullet list.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    channel=channel,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You turn the facts in the conversation into a short report.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    channel=channel,
)

task = "Report on the EV battery market."
researcher.run(task)
writer.run(task)

for message in channel.messages():
    print(f"\n## {message['agent_name']}\n{message['content']}")
