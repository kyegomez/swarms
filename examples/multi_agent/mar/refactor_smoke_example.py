"""Minimal example for `MultiAgentRouter`.

Demonstrates the list-based `self.agents` shape and that the boss agent
can route to any registered worker.
"""

from swarms import Agent
from swarms.structs.multi_agent_router import MultiAgentRouter


def main() -> None:
    research = Agent(
        agent_name="ResearchAgent",
        description=(
            "Researches topics in depth and returns factual summaries."
        ),
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )
    code = Agent(
        agent_name="CodeAgent",
        description=(
            "Writes and explains Python code with clean, idiomatic style."
        ),
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )
    writing = Agent(
        agent_name="WritingAgent",
        description="Drafts clear prose suitable for blog posts and reports.",
        model_name="gpt-5.4",
        max_loops=1,
        print_on=False,
    )

    router = MultiAgentRouter(
        name="demo-router",
        agents=[research, code, writing],
        model="gpt-5.4",
        print_on=False,
    )

    print(router)
    result = router.run(
        "Write a Python function that returns the nth Fibonacci number."
    )
    print("result:", result)


if __name__ == "__main__":
    main()
