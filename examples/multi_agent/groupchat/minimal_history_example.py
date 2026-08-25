
import json

from swarms import Agent
from swarms.structs.groupchat import GroupChat

MODEL = "gpt-5.4"


def agent(name: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=MODEL,
        max_loops=1,
        reasoning_effort=None,
    )


chat = GroupChat(
    agents=[
        agent("Optimist", "You argue the upside. One sentence."),
        agent("Skeptic", "You argue the risks. One sentence."),
    ],
    max_loops=4,
    threshold=0.3,
)

chat.run("Should small teams adopt multi-agent AI?")

messages = chat.conversation.return_messages_as_list()

print(json.dumps(messages, indent=4))
