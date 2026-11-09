"""Three agents talk through a QueueChannel; the main thread prints each message the moment it lands."""

import threading

from swarms import Agent, QueueChannel, SequentialWorkflow

channel = QueueChannel()

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

workflow = SequentialWorkflow(
    agents=[researcher, analyst, writer],
    max_loops=1,
    channel=channel,
)


def produce():
    try:
        workflow.run("Report on the EV battery market.")
    finally:
        # Wakes the consumer below and ends its loop.
        channel.close()


threading.Thread(target=produce, daemon=True).start()

# Blocks on each message instead of polling; returns when the producer closes the channel.
for message in channel:
    print(f"\n[{message['agent_name']}]\n{message['content']}")
