"""
All cache_config options at once — Anthropic.

Every key shown with its meaning. This is the fully-specified form; in practice
`prompt_caching=True` with a couple of keys is enough.

    export ANTHROPIC_API_KEY="sk-ant-..."
    python 6_all_options.py
"""

from swarms import Agent

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Ground claims in fundamentals; "
    "state assumptions; analysis only, never personalized advice. "
    * 200
)

agent = Agent(
    agent_name="Analyst",
    system_prompt=SYSTEM_PROMPT,
    model_name="claude-opus-4-8",
    prompt_caching=True,
    cache_config={
        "ttl": "1h",  # "5m" (default) | "1h"
        "cache_system_prompt": True,  # cache the system prefix
        "cache_messages": True,  # cache through the last message
        "cache_tools": True,  # cache the tool block (if tools are set)
        "override": None,  # None=auto; True/False forces injection
        "prompt_cache_key": None,  # OpenAI-only routing hint
        "prompt_cache_retention": None,  # OpenAI-only: "in_memory" | "24h"
    },
    max_loops=1,
    temperature=None,
)

print(
    agent.run("Summarize your standing instructions in one sentence.")
)
