import json

from swarms import Agent, MixtureOfAgents

MODEL = "gpt-5.4"


def agent(name: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=MODEL,
        max_loops=1,
    )


swarm = MixtureOfAgents(
    agents=[
        agent("Optimist", "Give one upside. One sentence."),
        agent("Pessimist", "Give one risk. One sentence."),
    ],
    aggregator_agent=agent(
        "Aggregator",
        "Combine the contributions into one verdict. Two sentences.",
    ),
    layers=2,
)

swarm.run("Should a small team adopt multi-agent AI?")

messages = swarm.conversation.return_messages_as_list()

print(json.dumps(messages, indent=4))
