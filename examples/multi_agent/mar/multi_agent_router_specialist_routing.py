from dotenv import load_dotenv

from swarms import Agent
from swarms.structs.multi_agent_router import MultiAgentRouter

load_dotenv()

# The boss routes on these descriptions, so they carry the weight here - the
# system_prompt only shapes the answer once an agent has been chosen.
researcher = Agent(
    agent_name="Researcher",
    agent_description="Finds and explains facts, prior art, and how something works. Use for 'what is', 'how does', and background questions.",
    system_prompt=(
        "You are a research analyst. Answer with concrete facts and the "
        "reasoning behind them. Keep it under 200 words."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)

coder = Agent(
    agent_name="Coder",
    agent_description="Writes and debugs code. Use for anything that should produce a function, script, or fix.",
    system_prompt=(
        "You are a Python engineer. Return working code with a short "
        "explanation. Prefer the standard library."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Turns material into clear prose for a stated audience. Use for summaries, docs, and explanations for non-experts.",
    system_prompt=(
        "You are a technical writer. Write plainly for a non-expert reader. "
        "No filler."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)

router = MultiAgentRouter(
    name="engineering-router",
    description="Routes engineering questions to the right specialist",
    agents=[researcher, coder, writer],
    model="gpt-5.4",
    # Print the boss's routing decision, including its reasoning.
    print_on=True,
    # Do not run an agent the boss handed an empty task.
    skip_null_tasks=True,
)

# One clear task, so the boss picks a single agent.
print("\n=== SINGLE AGENT ===")
single = router.run(
    "Write a Python function that validates an email address."
)
print(single)

# Several distinct pieces of work, so the boss picks several agents. They run
# concurrently, so this costs roughly one agent's latency rather than three.
print("\n=== MULTIPLE AGENTS ===")
multiple = router.run(
    "Explain what a vector database is, write a Python snippet that queries "
    "one, and summarise both for a non-technical reader."
)
print(multiple)

print("\n=== WHO ANSWERED ===")
for message in router.conversation.conversation_history:
    print(f"[{message['role']}] {str(message['content'])[:100]}")
