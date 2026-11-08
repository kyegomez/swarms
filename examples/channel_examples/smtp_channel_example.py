"""Two agents share an email thread in your own inbox; the people in `to` can reply.

Set SMTP_USERNAME and SMTP_PASSWORD first (an app password for Gmail).
Microsoft 365 is not supported: it no longer accepts passwords over IMAP.
"""

from swarms import Agent, SmtpChannel

channel = SmtpChannel(
    from_address="agents@yourco.com",
    to="you@yourco.com",
    smtp_host="smtp.yourco.com",
    imap_host="imap.yourco.com",
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

channel.close()
