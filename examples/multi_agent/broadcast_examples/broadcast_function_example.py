import asyncio

from swarms import Agent, broadcast

MODEL = "openrouter/z-ai/glm-5.3"

lead = Agent(
    agent_name="Tech-Lead",
    system_prompt="You describe a proposed architecture change in three sentences.",
    model_name=MODEL,
    max_loops=1,
    print_on=False,
)

reviewers = [
    Agent(
        agent_name=name,
        system_prompt=f"You are the {name} reviewer. In one sentence, give your verdict on the proposal.",
        model_name=MODEL,
        max_loops=1,
        print_on=False,
    )
    for name in ["Security", "Performance", "Cost"]
]


async def main():
    history = await broadcast(
        sender=lead,
        agents=reviewers,
        task="Propose moving the session store from Redis to Postgres.",
        output_type="dict",
    )
    for message in history:
        print(f"[{message['role']}] {message['content']}\n")


if __name__ == "__main__":
    asyncio.run(main())
