import json

from swarms import Agent, AgentRearrange

MODEL = "gpt-5.4"


def agent(name: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name=MODEL,
        max_loops=1,
    )


swarm = AgentRearrange(
    agents=[
        agent("Planner", "Give a one-line plan."),
        agent(
            "Coder",
            "Given the plan, describe the code in one sentence.",
        ),
        agent(
            "Reviewer", "Given the plan, name one risk. One sentence."
        ),
    ],
    # Planner first, then Coder and Reviewer concurrently.
    flow="Planner -> Coder, Reviewer",
    max_loops=1,
)

swarm.run("Build a function that validates email addresses.")

messages = swarm.conversation.return_messages_as_list()

print(json.dumps(messages, indent=4))
