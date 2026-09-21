from dotenv import load_dotenv

from swarms import Agent
from swarms.structs.ma_blocks import aggregate

load_dotenv()

MODEL = "gpt-5.5"


def worker(name: str, description: str, system_prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        agent_description=description,
        system_prompt=system_prompt,
        model_name=MODEL,
        max_loops=1,
        max_tokens=400,
        persistent_memory=False,
    )


workers = [
    worker(
        "Cost-Analyst",
        "Reasons about total cost of ownership.",
        "You analyse cost. Answer in at most 4 short bullets.",
    ),
    worker(
        "Reliability-Analyst",
        "Reasons about failure modes and operations.",
        "You analyse reliability and operational risk. Answer in at most 4 short bullets.",
    ),
    worker(
        "Developer-Experience-Analyst",
        "Reasons about the people who use the system daily.",
        "You analyse developer experience. Answer in at most 4 short bullets.",
    ),
]

task = "Should a 10-person startup run its own Postgres or use a managed one?"

# Explicit, so the example does not ride on whatever the library default is.
out = aggregate(
    workers=workers,
    task=task,
    type="dict",
    aggregator_model_name=MODEL,
)

print("\n=== recorded conversation ===")
for message in out:
    body = str(message["content"]).strip().replace("\n", " ")
    print(
        f"\n[{message['role']}] {len(str(message['content']))} chars"
    )
    print(f"  {body[:220]}")

print("\n=== roles in order ===")
print([m["role"] for m in out])

print("\n=== checks ===")
worker_turns = out[: len(workers)]
print(
    "each worker turn is an answer, not a transcript:",
    all(task not in str(m["content"]) for m in worker_turns),
)
print(
    "one turn per worker plus the aggregator:",
    [m["role"] for m in out]
    == [w.agent_name for w in workers] + ["Aggregator"],
)
print(
    "aggregator synthesised rather than echoing its prompt:",
    "comprehensive summary report"
    not in str(out[-1]["content"])[:80].lower()
    or len(str(out[-1]["content"])) > 2000,
)
