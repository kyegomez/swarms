"""
The three point-to-point communication patterns, as classes and as functions.

- OneToOne / one_to_one: a sender and a receiver exchange messages.
- Broadcast / broadcast: one sender, many receivers.
- OneToThree / one_to_three: one sender, exactly three receivers.

Requires an ANTHROPIC_API_KEY (or swap MODEL for any LiteLLM model).
"""

import asyncio

from dotenv import load_dotenv

from swarms import (
    Agent,
    Broadcast,
    OneToOne,
    OneToThree,
    broadcast,
    one_to_one,
    one_to_three,
)

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"


def make_agent(name: str, role: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=f"You are {name}, {role}. Answer in one sentence.",
        model_name=MODEL,
        max_loops=1,
        print_on=False,
    )


def show(title: str, history: list) -> None:
    print(f"\n=== {title} ===")
    for message in history:
        print(f"[{message['role']}] {message['content']}")


if __name__ == "__main__":
    task = "Propose one name for a new open-source vector database."

    sender = make_agent("Proposer", "a product namer")
    critic = make_agent("Critic", "a blunt reviewer of names")
    reviewers = [
        make_agent("Marketing", "a marketing lead"),
        make_agent("Legal", "a trademark lawyer"),
        make_agent("Engineer", "a backend engineer"),
    ]

    # One-to-one, as a class and as a function.
    pair = OneToOne(sender, critic, output_type="dict")
    show("OneToOne class", pair.run(task))
    show("one_to_one function", one_to_one(sender, critic, task))

    # Broadcast, as a class (sync) and as a function (async).
    group = Broadcast(sender, reviewers, output_type="dict")
    show("Broadcast class", group.run(task))
    show(
        "broadcast function",
        asyncio.run(broadcast(sender, reviewers, task)),
    )

    # One-to-three, as a class and as a function.
    trio = OneToThree(sender, reviewers, output_type="dict")
    show("OneToThree class", trio.run(task))
    show(
        "one_to_three function", one_to_three(sender, reviewers, task)
    )
