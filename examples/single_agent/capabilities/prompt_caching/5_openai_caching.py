"""
Prompt caching — OpenAI.

OpenAI caches automatically for prompts >= 1,024 tokens (no cache_control
markers). `cache_config` passes through OpenAI's two optional controls:
  * prompt_cache_key       — routing hint that raises cache hit rates
  * prompt_cache_retention — "in_memory" (default) or "24h"

    export OPENAI_API_KEY="sk-..."
    python 5_openai_caching.py
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
    model_name="gpt-5.4",
    prompt_caching=True,
    cache_config={
        "prompt_cache_key": "analyst-v1",
        "prompt_cache_retention": "24h",
    },
    max_loops=1,
)

print(
    agent.run(
        "Give me a 3-point framework for valuing a SaaS company."
    )
)
print(
    agent.run("Now do the same for a bank.")
)  # hits the automatic cache
