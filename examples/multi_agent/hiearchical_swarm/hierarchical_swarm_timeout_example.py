"""
HierarchicalSwarm — worker timeout & retry demo

Market-Analyst is patched to hang indefinitely, simulating a stuck MCP/tool call.
worker_timeout=90s fires after 90 seconds, marks the agent [FAILED], and
Growth-Strategist finishes normally with a real claude-sonnet-4-5 response —
no hung swarm.
"""

import sys
import io
import time

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace"
)
sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer, encoding="utf-8", errors="replace"
)

from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

analyst = Agent(
    agent_name="Market-Analyst",
    agent_description="Analyses market trends",
    system_prompt="You are a senior market analyst. Be concise — 3 bullets max.",
    model_name="claude-sonnet-4-5",
    max_tokens=8000,
    thinking_tokens=4000,
    temperature=1,
    max_loops=1,
    verbose=False,
    print_on=False,
)

strategist = Agent(
    agent_name="Growth-Strategist",
    agent_description="Turns market insights into growth plans",
    system_prompt="You are a growth strategist. Propose 3 concrete initiatives.",
    model_name="claude-sonnet-4-5",
    max_tokens=8000,
    thinking_tokens=4000,
    temperature=1,
    max_loops=1,
    verbose=False,
    print_on=False,
)

swarm = HierarchicalSwarm(
    name="Timeout-Demo-Swarm",
    description="Market analysis and strategy team.",
    agents=[analyst, strategist],
    max_loops=1,
    director_model_name="claude-sonnet-4-5",
    director_temperature=1,
    director_top_p=None,
    planning_enabled=False,
    director_feedback_on=False,
    autosave=False,
    verbose=False,
    worker_timeout=90,
    heartbeat_interval=30,
    max_retries=0,
)

_original_call = swarm.call_single_agent


def _patched_call(
    agent_name,
    task,
    streaming_callback=None,
    _add_to_conversation=True,
):
    if agent_name == "Market-Analyst":
        print(
            "  [demo] Market-Analyst hanging (simulating stuck tool)..."
        )
        time.sleep(3600)
    return _original_call(
        agent_name, task, streaming_callback, _add_to_conversation
    )


swarm.call_single_agent = _patched_call

result = swarm.run(
    "Competitive snapshot of the async-meeting-AI market for a new B2B SaaS launch."
)
print(result)
