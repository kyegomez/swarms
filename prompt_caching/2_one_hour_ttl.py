"""
1-hour cache TTL — Anthropic.

By default the cache lives ~5 minutes. `cache_config={"ttl": "1h"}` opts into
Anthropic's 1-hour cache (survives longer gaps; 2x write cost). The required
extended-cache beta header is attached automatically.

    export ANTHROPIC_API_KEY="sk-ant-..."
    python 2_one_hour_ttl.py
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
    cache_config={"ttl": "1h"},  # extended cache
    max_loops=1,
    temperature=None,
)

print(agent.run("Summarize your standing instructions in one line."))
