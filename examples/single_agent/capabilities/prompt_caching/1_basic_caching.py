"""
Basic prompt caching with an Agent.

`prompt_caching=True` is the only flag needed. On Anthropic it caches the large,
stable system prompt so repeat calls reuse it at a steep discount.

    export ANTHROPIC_API_KEY="sk-ant-..."
    python 1_basic_caching.py
"""

from swarms import Agent

# Large, stable prefix (repeated to clear Opus 4.x's 4,096-token cache minimum).
SYSTEM_PROMPT = (
    "You are a senior financial analyst. Ground every claim in fundamentals, "
    "state assumptions explicitly, and provide analysis only — never "
    "personalized investment advice. " * 200
)

agent = Agent(
    agent_name="Analyst",
    system_prompt=SYSTEM_PROMPT,
    model_name="claude-opus-4-8",
    prompt_caching=True,  # <-- the on-switch
    max_loops=1,
    temperature=None,  # Opus 4.8 rejects a temperature value
)

print(
    agent.run(
        "Give me a 3-point framework for valuing a SaaS company."
    )
)
print(
    agent.run("Now do the same for a bank.")
)  # reuses the cached prefix
