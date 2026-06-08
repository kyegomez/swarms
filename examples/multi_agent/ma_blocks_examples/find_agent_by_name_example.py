"""Minimal example for `find_agent_by_name`.

Builds a small list of agents and looks them up by name. Demonstrates that
both the `agent_name` and `name` attributes are honored, and that repeated
lookups hit the internal cache.
"""

from swarms import Agent
from swarms.structs.ma_blocks import find_agent_by_name


def build_agents() -> list:
    return [
        Agent(
            agent_name="Researcher",
            model_name="gpt-4.1",
            max_loops=1,
            print_on=False,
        ),
        Agent(
            agent_name="Writer",
            model_name="gpt-4.1",
            max_loops=1,
            print_on=False,
        ),
        Agent(
            agent_name="Editor",
            model_name="gpt-4.1",
            max_loops=1,
            print_on=False,
        ),
    ]


def main() -> None:
    agents = build_agents()

    researcher = find_agent_by_name(agents, "Researcher")
    print(f"Found by agent_name: {researcher.agent_name}")

    # Second lookup hits the cached index.
    writer = find_agent_by_name(agents, "Writer")
    print(f"Found (cached): {writer.agent_name}")

    try:
        find_agent_by_name(agents, "Nonexistent")
    except ValueError as exc:
        print(f"Not-found path OK: {exc}")


if __name__ == "__main__":
    main()
