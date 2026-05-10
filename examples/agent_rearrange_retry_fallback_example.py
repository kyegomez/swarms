"""
AgentRearrange — per-node retry & fallback annotations
=======================================================
  "A -> B!3 -> C"       B retries up to 3 times on failure
  "A -> B!3>D -> C"     B retries 3 times, then falls back to D
  "A -> B?D -> C"       B routes to D on first failure

Each example patches the middle agent to fail a controlled number of times
so the retry and fallback behaviour is visible in the logs.

Model: claude-sonnet-4-5  |  Thinking enabled  |  Temperature 1.0
"""

from unittest.mock import patch, MagicMock
from swarms import Agent, AgentRearrange

MODEL      = "claude-sonnet-4-5"
TEMP       = 1.0
THINK      = 8000
MAX_TOKENS = 16000


def make_agent(name: str, system_prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=system_prompt,
        model_name=MODEL,
        temperature=TEMP,
        thinking_tokens=THINK,
        max_tokens=MAX_TOKENS,
        reasoning_enabled=True,
        max_loops=1,
        autosave=False,
        print_on=True,
    )


researcher = make_agent(
    "Researcher",
    "You are a research analyst. Given a topic, return 3 concise bullet "
    "points of key facts.",
)
fact_checker = make_agent(
    "FactChecker",
    "You are a fact-checker. Reply VERIFIED if the brief looks sound.",
)
fallback_summarizer = make_agent(
    "FallbackSummarizer",
    "You are a backup summarizer. Summarize the brief in 2 sentences, "
    "noting fact-checking was skipped.",
)
writer = make_agent(
    "Writer",
    "You are a writer. Turn the brief into a polished 100-word paragraph.",
)

TASK = "The impact of large language models on software engineering productivity"


# ── Example 1: B!3 — retry fires twice then succeeds ─────────────────────────
print("\n" + "=" * 70)
print("EXAMPLE 1 — FactChecker!3: fails twice, succeeds on 3rd attempt")
print("=" * 70)

_call_count = 0
_real_run = fact_checker.run

def _fail_twice(task, **kwargs):
    global _call_count
    _call_count += 1
    if _call_count < 3:
        raise RuntimeError(f"FactChecker simulated failure (attempt {_call_count})")
    return _real_run(task=task, **kwargs)

fact_checker.run = _fail_twice

pipeline1 = AgentRearrange(
    name="RetryPipeline",
    agents=[researcher, fact_checker, writer],
    flow="Researcher -> FactChecker!3 -> Writer",
    max_loops=1,
    output_type="final",
)
result1 = pipeline1.run(TASK)
fact_checker.run = _real_run  # restore
print("\n[Output]\n", result1)


# ── Example 2: B!1>D — retries once, exhausts, falls back to FallbackSummarizer
print("\n" + "=" * 70)
print("EXAMPLE 2 — FactChecker!1>FallbackSummarizer: always fails, fallback runs")
print("=" * 70)

def _always_fail(task, **kwargs):
    raise RuntimeError("FactChecker permanently down")

fact_checker.run = _always_fail

pipeline2 = AgentRearrange(
    name="RetryFallbackPipeline",
    agents=[researcher, fact_checker, fallback_summarizer, writer],
    flow="Researcher -> FactChecker!1>FallbackSummarizer -> Writer",
    max_loops=1,
    output_type="final",
)
result2 = pipeline2.run(TASK)
fact_checker.run = _real_run  # restore
print("\n[Output]\n", result2)


# ── Example 3: B?D — immediate fallback on first failure ─────────────────────
print("\n" + "=" * 70)
print("EXAMPLE 3 — FactChecker?FallbackSummarizer: routes to fallback immediately")
print("=" * 70)

fact_checker.run = _always_fail

pipeline3 = AgentRearrange(
    name="ImmediateFallbackPipeline",
    agents=[researcher, fact_checker, fallback_summarizer, writer],
    flow="Researcher -> FactChecker?FallbackSummarizer -> Writer",
    max_loops=1,
    output_type="final",
)
result3 = pipeline3.run(TASK)
fact_checker.run = _real_run  # restore
print("\n[Output]\n", result3)
