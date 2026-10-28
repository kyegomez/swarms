"""Two agents share a conversation kept on another server, reached over HTTP.

Start the server first in another terminal:
    python examples/channel_examples/http_channel_server.py
"""

from swarms import Agent, HttpChannel

URL = "http://127.0.0.1:8765/channels/ev-report"

# Each agent gets its own client, as if it ran on a different machine.
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You gather facts and figures. Answer in a short bullet list.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    channel=HttpChannel(URL),
)

writer = Agent(
    agent_name="Writer",
    system_prompt="You turn the facts in the conversation into a short report.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    channel=HttpChannel(URL),
)

task = "Report on the EV battery market."
researcher.run(task)
writer.run(task)

for message in HttpChannel(URL).messages():
    print(f"\n## {message['agent_name']}\n{message['content']}")
