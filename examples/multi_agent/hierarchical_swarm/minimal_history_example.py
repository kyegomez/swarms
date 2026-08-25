

from swarms import Agent, HierarchicalSwarm
import json

MODEL = "gpt-5.4"


def agent(name: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=MODEL,
        max_loops=1,
        reasoning_effort=None,
    )


swarm = HierarchicalSwarm(
    agents=[
        agent("Historian", "Give dates and facts. One sentence."),
        agent("Economist", "Give economic reasoning. One sentence."),
    ],
    director_model_name=MODEL,
    director_settings={"reasoning_effort": None},
    max_loops=1,
)

out = swarm.run("Explain the 2022 interest rate hikes.")


messages = swarm.conversation.return_messages_as_list()

print(json.dumps(messages, indent=4))
