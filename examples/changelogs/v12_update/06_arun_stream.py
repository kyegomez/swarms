import asyncio
import sys
from swarms import Agent

agent = Agent(
    agent_name="StreamingAgent",
    agent_description="Streams tokens asynchronously to the caller",
    model_name="gpt-5.4-mini",
    max_loops=1,
    streaming_on=True,
    persistent_memory=False,
)


async def stream_to_console(task: str) -> str:
    """Stream an agent response token-by-token, collecting the full text."""
    full_response = []
    print(f"Task: {task}\nResponse: ", end="", flush=True)
    async for token in agent.arun_stream(task):
        sys.stdout.write(token)
        sys.stdout.flush()
        full_response.append(token)
    print()  # newline after stream ends
    return "".join(full_response)


async def main():
    # --- Single streaming call ---
    print("=== arun_stream: single call ===")
    response = await stream_to_console(
        "Explain the difference between concurrency and parallelism in two sentences."
    )
    print(f"\n[Total chars: {len(response)}]")

    # --- Multiple concurrent streaming calls ---
    print("\n=== arun_stream: two concurrent tasks ===")

    async def collect(task: str, label: str) -> None:
        tokens = []
        async for token in agent.arun_stream(task):
            tokens.append(token)
        print(f"[{label}] {len(tokens)} tokens received")

    await asyncio.gather(
        collect("What is asyncio?", "TaskA"),
        collect("What is a thread pool?", "TaskB"),
    )

    # --- Exception propagation demo ---
    print("\n=== arun_stream: exception propagation ===")
    bad_agent = Agent(
        agent_name="BadAgent",
        model_name="gpt-5.4-mini",
        max_loops=1,
        persistent_memory=False,
    )
    try:
        # Force an error by passing None — exception surfaces to the caller
        async for _ in bad_agent.arun_stream(None):  # type: ignore[arg-type]
            pass
    except Exception as exc:
        print(
            f"Exception correctly propagated: {type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    asyncio.run(main())
