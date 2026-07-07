"""
Prompt caching — full `cache_config` example
============================================

`prompt_caching=True` is the on-switch. `cache_config={...}` tunes how caching
happens; every key is optional with a sensible default. This example sets them
all on one Anthropic agent, then verifies real cache activity via `usage`.

    export ANTHROPIC_API_KEY="sk-ant-..."
    python cache_config_example.py
"""

from swarms import Agent

# A large, STABLE prefix — repeated to clear Opus 4.x's 4,096-token cache
# minimum. In a real app this is your policy docs / long context / schema.
KNOWLEDGE_BASE = (
    "You are a senior financial analyst. Standing instructions: "
    + (
        "Ground claims in fundamentals — revenue growth, margins, free cash "
        "flow, balance-sheet health, and moat. State assumptions explicitly. "
        "Prefer ranges over false precision. Analysis only; never give "
        "personalized investment advice. "
    )
    * 20
)

agent = Agent(
    agent_name="CachedAnalyst",
    system_prompt=KNOWLEDGE_BASE,
    model_name="claude-opus-4-8",
    prompt_caching=True,  # <-- the on-switch
    cache_config={
        "ttl": "1h",  # "5m" (default) | "1h" extended Anthropic cache
        "cache_system_prompt": True,  # cache the large stable system prefix
        "cache_messages": True,  # cache through the last message (multi-turn)
        "cache_tools": True,  # cache the tool-definitions block
        "override": None,  # None=auto (Anthropic->inject); True/False forces it
        "prompt_cache_key": None,  # OpenAI-only: routing hint for hit rate
        "prompt_cache_retention": None,  # OpenAI-only: "in_memory" | "24h"
    },
    max_loops=1,
    max_tokens=64,
    temperature=None,  # Opus 4.8 rejects a temperature value
    persistent_memory=False,
    print_on=False,
)

# First call writes the cached prefix; second call reads it back at a discount.
print("=== call 1 (cache write) ===")
print(
    agent.run(
        "State your single most important instruction in one sentence."
    )
)

print("\n=== call 2 (cache hit) ===")
print(agent.run("Now restate it as a checklist item."))

# Confirm caching actually happened. Agent.run() returns a formatted string, so
# to see the token usage we read it from the agent's OWN underlying llm
# (agent.llm) — the exact instance the Agent built from cache_config above.
# Look for cache_creation_input_tokens (write) then cache_read_input_tokens (read).
print("\n=== usage (proof of caching) ===")
agent.llm.return_all = (
    True  # expose the raw response so we can read `usage`
)
for label in ("write", "read"):
    resp = agent.llm.run("Restate your key instruction.")
    usage = resp["usage"] if isinstance(resp, dict) else resp.usage
    print(f"[{label}] {usage}")
