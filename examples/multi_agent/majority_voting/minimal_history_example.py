

import json

from swarms import Agent, MajorityVoting

MODEL = "gpt-5.4"

swarm = MajorityVoting(
    agents=[
        Agent(
            agent_name=f"Voter-{i}",
            system_prompt="Answer with one word: PYTHON or RUST.",
            model_name=MODEL,
            max_loops=1,
        )
        for i in range(3)
    ],
    consensus_agent_model_name=MODEL,
    max_loops=1,
)

swarm.run("Faster for a high-performance web server: PYTHON or RUST?")

messages = swarm.conversation.return_messages_as_list()

print(json.dumps(messages, indent=4))
