"""
Every cache_config option in one Agent.

Each key is shown with its meaning. In practice `prompt_caching=True` with one
or two keys is enough — this is the fully-specified reference form.

    export ANTHROPIC_API_KEY="sk-ant-..."
    python 6_cache_config_all_options.py
"""

from swarms import Agent

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Ground every claim in fundamentals, "
    "state assumptions explicitly, and provide analysis only — never "
    "personalized investment advice. " * 200
)

agent = Agent(
    agent_name="Analyst",
    system_prompt=SYSTEM_PROMPT,
    model_name="claude-opus-4-8",
    prompt_caching=True,  # master on-switch
    cache_config={
        "ttl": "1h",  # "5m" (default) | "1h" extended cache
        "cache_system_prompt": True,  # cache the stable system prefix
        "cache_messages": True,  # cache through the last message (multi-turn)
        "cache_tools": True,  # cache the tool block (when tools are set)
        "override": None,  # None=auto (Anthropic->inject); True/False forces it
        "prompt_cache_key": None,  # OpenAI-only: routing hint
        "prompt_cache_retention": None,  # OpenAI-only: "in_memory" | "24h"
    },
    max_loops=1,
    temperature=None,
)

print(
    agent.run("Summarize your standing instructions in one sentence.")
)
