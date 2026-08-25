import json

from swarms import Agent, SwarmRouter

MODEL = "gpt-5.4"


def agent(name: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=MODEL,
        max_loops=1,
    )


router = SwarmRouter(
    agents=[
        agent("Scout", "Give one obscure fact. One sentence."),
        agent(
            "Builder",
            "Build one idea on the fact you were given. One sentence.",
        ),
    ],
    swarm_type="SequentialWorkflow",
    multi_agent_collab_prompt=True,
    max_loops=1,
)

router.run("Begin.")

# The router builds its swarm lazily on the first run, so read the history
# from that swarm afterwards.
conversation = router.swarm.agent_rearrange.conversation

messages = conversation.return_messages_as_list()

print(json.dumps(messages, indent=4))
