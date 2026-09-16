"""Seed an agent with prior turns: at construction, and per run."""

from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

prior_turns = [
    {
        "role": "user",
        "content": "My project is called Helios and it is a solar forecasting service.",
    },
    {
        "role": "assistant",
        "content": "Noted. Helios forecasts solar output.",
    },
]

# Constructor messages: seeded into short_memory, re-sent on every run.
agent = Agent(
    agent_name="Messages-Example-Agent",
    model_name="gpt-4.1-mini",
    max_loops=1,
    messages=prior_turns,
    persistent_memory=False,
)

print("=== conversation seeded at construction ===")
for message in agent.short_memory.conversation_history:
    print(f"{message['role']}: {str(message['content'])[:80]}")

out = agent.run("What is my project called, and what does it do?")
print("\n=== run() output ===")
print(out)

# Per-call messages: added to the conversation, then sent as this turn's context.
fresh = Agent(
    agent_name="Messages-Per-Call-Agent",
    model_name="gpt-4.1-mini",
    max_loops=1,
    persistent_memory=False,
)

out = fresh.run(
    "Which of those two numbers was larger?",
    messages=[
        {
            "role": "user",
            "content": "Remember these numbers: 47 and 12.",
        },
        {"role": "assistant", "content": "Got it: 47 and 12."},
    ],
)
print("\n=== run(messages=...) output ===")
print(out)

print("\n=== conversation after run(messages=...) ===")
for message in fresh.short_memory.conversation_history:
    print(f"{message['role']}: {str(message['content'])[:80]}")
