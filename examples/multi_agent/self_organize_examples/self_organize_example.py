"""
Minimal SelfOrganizingSwarm example.

Three agents each hold a private number. They must message each other
to collect every number and return them sorted. Requires communication.

Usage:
    export OPENAI_API_KEY=...
    python examples/multi_agent/self_organize_examples/self_organize_example.py
"""

import json

from dotenv import load_dotenv

from swarms import SelfOrganizingSwarm

load_dotenv()

PRIVATE_NUMBERS = [7, 42, 19]
TARGET = sorted(PRIVATE_NUMBERS)


def is_correct(answer: str) -> bool:
    try:
        return json.loads(answer) == TARGET
    except Exception:
        return False


swarm = SelfOrganizingSwarm(
    num_agents=len(PRIVATE_NUMBERS),
    model_name="gpt-5.4",
    max_tokens_per_agent=2048,
    timeout=120.0,
    verbose=True,
)

result = swarm.run(
    task=(
        f"{len(PRIVATE_NUMBERS)} agents each have one private number. "
        "Share your number with the team, collect every number, and "
        "submit them sorted ascending. Final format: #### followed by "
        f"a JSON array of {len(PRIVATE_NUMBERS)} integers, "
        "e.g. #### [1, 2, 3]"
    ),
    private_inputs=[
        f"Your private number is {n}." for n in PRIVATE_NUMBERS
    ],
    verifier=is_correct,
    rules=(
        "Tell teammates your number. Read their messages. Submit only "
        "when you have one number from every agent, including yourself."
    ),
)

print("answer:", result["answer"])
print("success:", result["success"])
print("solved_by:", result["solved_by"])
print("team_reward:", result["team_reward"])
print("tokens:", result["total_tokens"])
print("messages:", result["message_calls"])
print("elapsed:", result["elapsed_seconds"], "s")
