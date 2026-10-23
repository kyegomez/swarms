"""A self-selecting chat whose accepted messages become the file."""

from swarms import Agent, FileChannel
from swarms.structs.groupchat import RESPOND_TOOL, GroupChat


def speaker(name, stance):
    return Agent(
        agent_name=name,
        system_prompt=f"You argue for {stance}. Keep each message short.",
        model_name="gpt-5.4",
        max_loops=1,
        persistent_memory=False,
        tools_list_dictionary=[RESPOND_TOOL],
    )


channel = FileChannel("runs/groupchat.md")

chat = GroupChat(
    agents=[
        speaker("Optimist", "the benefits"),
        speaker("Pessimist", "the risks"),
        speaker("Realist", "a balanced view"),
    ],
    max_loops=6,
    threshold=0.5,
    idle_timeout=8.0,
    channel=channel,
)
chat.run("Should hospitals adopt AI for triage?")

print(channel.path.read_text())
