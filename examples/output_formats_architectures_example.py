"""Compact examples demonstrating `Agent` output types and two architectures.

These examples are illustrative — real runs require LLM credentials and full configuration.
"""

from typing import Any

from swarms.structs.agent import Agent


def simple_agent_examples() -> None:
    # Example: final-only output
    agent_final = Agent(llm="gpt-4o-mini", output_type="final")
    resp = agent_final.run("Give a one-sentence summary of climate change.")
    print("Final-only response:", resp)

    # Example: full history as list
    agent_list = Agent(llm="gpt-4o-mini", output_type="list")
    resp_list = agent_list.run("Explain reinforcement learning in simple terms.")
    print("History (list):", resp_list)


def demo_streaming() -> None:
    # Token streaming example
    def stream_cb(token: str) -> None:
        print(token, end="", flush=True)

    agent_stream = Agent(
        llm="gpt-4o-mini",
        stream=True,
        streaming_on=True,
        streaming_callback=stream_cb,
        output_type="final",
    )
    print("Streaming generation:")
    agent_stream.run("Write a motivational haiku about focus.")


def sequential_workflow_example() -> None:
    from swarms.structs.sequential_workflow import SequentialWorkflow

    # Compose two simple agents (illustrative): agent A prepares, agent B finalizes
    agent_a = Agent(llm="gpt-4o-mini", output_type="dict-final")
    agent_b = Agent(llm="gpt-4o-mini", output_type="final")

    seq = SequentialWorkflow(agents=[agent_a, agent_b])
    out = seq.run("Create a short marketing blurb for a new smartwatch.")
    print("Sequential workflow output:", out)


def concurrent_workflow_example() -> None:
    from swarms.structs.concurrent_workflow import ConcurrentWorkflow

    workers = [Agent(llm="gpt-4o-mini", output_type="dict") for _ in range(3)]
    cw = ConcurrentWorkflow(agents=workers)
    result = cw.run("Analyze customer feedback and return top 3 themes.")
    print("Concurrent workflow result:", result)


if __name__ == "__main__":
    simple_agent_examples()
    demo_streaming()
    sequential_workflow_example()
    concurrent_workflow_example()
