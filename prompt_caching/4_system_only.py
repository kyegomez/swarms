"""
Cache only the system prompt — Anthropic.

By default both the system prompt and the last message get a cache breakpoint.
Set `cache_messages=False` to cache only the stable system prefix (useful when
each user turn is different and won't be reused).

    export ANTHROPIC_API_KEY="sk-ant-..."
    python 4_system_only.py
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
        "cache_system_prompt": True,
        "cache_messages": False,  # don't cache the (changing) user turn
    },
    max_loops=1,
    temperature=None,
)

print(agent.run("List one risk for an EV maker."))
