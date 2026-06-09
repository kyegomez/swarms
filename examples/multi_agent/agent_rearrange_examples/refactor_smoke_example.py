"""Minimal example for `AgentRearrange`.

Exercises the list-based `self.agents` (post-refactor) with a flow that
mixes sequential and concurrent execution.
"""

from swarms import Agent
from swarms.structs.agent_rearrange import AgentRearrange


def main() -> None:
    planner = Agent(
        agent_name="Planner",
        model_name="gpt-4.1",
        max_loops=1,
        print_on=False,
    )
    coder = Agent(
        agent_name="Coder",
        model_name="gpt-4.1",
        max_loops=1,
        print_on=False,
    )
    reviewer = Agent(
        agent_name="Reviewer",
        model_name="gpt-4.1",
        max_loops=1,
        print_on=False,
    )
    tester = Agent(
        agent_name="Tester",
        model_name="gpt-4.1",
        max_loops=1,
        print_on=False,
    )

    pipeline = AgentRearrange(
        name="plan-code-review-test",
        agents=[planner, coder, reviewer, tester],
        flow="Planner -> Coder -> Reviewer, Tester",
        max_loops=1,
    )

    # Mutation paths the refactor touched.
    extra = Agent(
        agent_name="Documenter",
        model_name="gpt-4.1",
        max_loops=1,
        print_on=False,
    )
    pipeline.add_agent(extra)
    print(
        "agents after add:",
        [a.agent_name for a in pipeline.agents],
    )
    pipeline.remove_agent("Documenter")
    print(
        "agents after remove:",
        [a.agent_name for a in pipeline.agents],
    )

    result = pipeline.run(
        "Build a Python function that validates an email address."
    )
    print("result:", result)


if __name__ == "__main__":
    main()
