import asyncio

from swarms import Agent


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city.

    Returns:
        A short weather report.
    """
    return f"{city}: 18C, light rain"


async def on_token(token: str) -> None:
    print(token, end="", flush=True)


async def main():
    agent = Agent(
        agent_name="Async-Analyst",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )

    # 1. Await a run on the event loop
    answer = await agent.arun("What is the capital of France?")
    print(answer)

    # 2. Stream tokens to a sync or async callback as they arrive
    await agent.arun(
        "Write a haiku about event loops.",
        streaming_callback=on_token,
    )
    print()

    # 3. Stream tokens with an async generator
    async for token in agent.arun_stream(
        "Explain async/await in two sentences."
    ):
        print(token, end="", flush=True)
    print()

    # 4. Tools work the same as with run
    weather_agent = Agent(
        agent_name="Weather-Agent",
        model_name="gpt-5.4",
        tools=[get_weather],
        max_loops=2,
        print_on=False,
    )
    print(await weather_agent.arun("What is the weather in Paris?"))

    # 5. Run many agents concurrently on one thread
    agents = [
        Agent(
            agent_name=f"Worker-{i}",
            model_name="gpt-5.4-mini",
            max_loops=1,
            print_on=False,
        )
        for i in range(5)
    ]
    results = await asyncio.gather(
        *(a.arun("Give one use case for AI agents.") for a in agents)
    )
    for a, result in zip(agents, results):
        print(f"{a.agent_name}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
