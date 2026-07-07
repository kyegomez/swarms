"""
Prompt caching test suite — all cache_config params, via Agent (agent.py)
=========================================================================

Exercises the ``prompt_caching`` switch and every ``cache_config`` option,
driven through ``Agent`` so it tests the real path:
Agent -> llm_handling() -> LiteLLM -> the provider request.

Two layers:

1. OFFLINE (always runs, no keys): builds Agents with each cache_config and
   inspects the request the Agent-built llm would send — cache_control markers
   on system / last message / tools, the TTL value, the override, and the
   OpenAI passthrough params. Precisely verifies every knob.

2. LIVE (needs keys): runs the full ``agent.run()`` path for Anthropic and
   OpenAI with representative configs and confirms real cache activity via
   ``usage``.

Run:
    export ANTHROPIC_API_KEY=...   # Claude
    export OPENAI_API_KEY=...      # GPT
    python test_prompt_caching.py

    # offline assertions only, no network:
    python test_prompt_caching.py --offline
"""

import os
import sys

from swarms import Agent

ANTHROPIC_MODEL = "claude-opus-4-8"
OPENAI_MODEL = "gpt-5.4"


# A large, STABLE prefix — repeated to clear Opus 4.x's 4,096-token minimum.
SYSTEM_PROMPT = (
    "You are a meticulous financial-analysis assistant. Standing instructions: "
    + (
        "Ground every claim in fundamentals — revenue growth, margins, free cash "
        "flow, balance-sheet health, and competitive moat. State assumptions "
        "explicitly. Prefer ranges over false precision. Provide analysis only; "
        "never give personalized investment advice. "
    )
    * 160
)

TASK = "In one sentence, restate your single most important standing instruction."

# Tool schemas, used to exercise cache_tools.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get a stock quote.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_filing",
            "description": "Fetch an SEC filing.",
            "parameters": {
                "type": "object",
                "properties": {"cik": {"type": "string"}},
                "required": ["cik"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_agent(model, cache_config=None, prompt_caching=True):
    """Build an Agent with caching configured. This is the object under test."""
    return Agent(
        agent_name="CacheTester",
        system_prompt=SYSTEM_PROMPT,
        model_name=model,
        prompt_caching=prompt_caching,
        cache_config=cache_config,
        max_loops=1,
        max_tokens=64,
        temperature=None,  # Opus 4.8 rejects a temperature value
        persistent_memory=False,
        print_on=False,
        verbose=False,
    )


def _cache_control_of(message):
    """Return the cache_control dict on a message's content, or None."""
    content = message.get("content")
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and "cache_control" in b:
                return b["cache_control"]
    return None


def _system_and_last(agent):
    """The system message and the last message of the Agent-built request."""
    msgs = agent.llm._prepare_messages(task=TASK)
    system = next(
        (m for m in msgs if m.get("role") == "system"), None
    )
    return system, msgs[-1]


def _has_key(env_names):
    return any(os.getenv(name) for name in env_names)


def _as_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return getattr(obj, "__dict__", {}) or {}


def extract_cache_metrics(usage):
    u = _as_dict(usage)
    created = u.get("cache_creation_input_tokens") or 0
    read = u.get("cache_read_input_tokens") or 0
    details = _as_dict(u.get("prompt_tokens_details"))
    cached = details.get("cached_tokens") or 0
    return {
        "created": int(created or 0),
        "read": int(read or 0),
        "cached": int(cached or 0),
    }


_SKIP_ERROR_HINTS = (
    "rate limit",
    "ratelimit",
    "quota",
    "insufficient_quota",
    "authentication",
    "invalid api key",
    "permission",
    "could not resolve authentication",
    "not have access",
)


def _is_skippable_error(exc):
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if (
        "ratelimit" in name
        or "authentication" in name
        or "permission" in name
    ):
        return True
    return any(hint in msg for hint in _SKIP_ERROR_HINTS)


# ---------------------------------------------------------------------------
# Tiny assertion harness
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label, ok):
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        return ok


# ---------------------------------------------------------------------------
# 1. OFFLINE — assert each cache_config knob shapes the request correctly
# ---------------------------------------------------------------------------
def run_offline_checks():
    print("=" * 72)
    print(
        "OFFLINE — cache_config knobs (Agent -> LiteLLM request, no network)"
    )
    print("=" * 72)
    c = Checker()

    # --- Anthropic default (prompt_caching=True, no cache_config) ---
    a = make_agent(ANTHROPIC_MODEL)
    sysmsg, last = _system_and_last(a)
    c.check(
        "anthropic default: system prompt cached",
        _cache_control_of(sysmsg) is not None,
    )
    c.check(
        "anthropic default: last message cached",
        _cache_control_of(last) is not None,
    )
    c.check(
        "anthropic default: TTL is 5m (no ttl key)",
        _cache_control_of(sysmsg) == {"type": "ephemeral"},
    )

    # --- ttl="1h" -> marker carries ttl + beta header attached ---
    a = make_agent(ANTHROPIC_MODEL, {"ttl": "1h"})
    sysmsg, _ = _system_and_last(a)
    c.check(
        "ttl=1h: marker carries ttl=1h",
        (_cache_control_of(sysmsg) or {}).get("ttl") == "1h",
    )
    cp = {}
    a.llm._apply_cache_request_params(cp)
    c.check(
        "ttl=1h: extended-cache beta header attached",
        (cp.get("extra_headers") or {}).get("anthropic-beta")
        == "extended-cache-ttl-2025-04-11",
    )

    # --- cache_system_prompt=False -> system NOT cached, last still cached ---
    a = make_agent(ANTHROPIC_MODEL, {"cache_system_prompt": False})
    sysmsg, last = _system_and_last(a)
    c.check(
        "cache_system_prompt=False: system NOT cached",
        _cache_control_of(sysmsg) is None,
    )
    c.check(
        "cache_system_prompt=False: last message still cached",
        _cache_control_of(last) is not None,
    )

    # --- cache_messages=False -> system cached, last NOT cached ---
    a = make_agent(ANTHROPIC_MODEL, {"cache_messages": False})
    sysmsg, last = _system_and_last(a)
    c.check(
        "cache_messages=False: system still cached",
        _cache_control_of(sysmsg) is not None,
    )
    c.check(
        "cache_messages=False: last message NOT cached",
        _cache_control_of(last) is None,
    )

    # --- cache_tools: last tool marked / not marked ---
    a = make_agent(ANTHROPIC_MODEL, {"cache_tools": True})
    cached_tools = a.llm._maybe_cache_tools(TOOLS)
    c.check(
        "cache_tools=True: last tool cached",
        "cache_control" in cached_tools[-1],
    )
    c.check(
        "cache_tools=True: original TOOLS untouched",
        "cache_control" not in TOOLS[-1],
    )
    a = make_agent(ANTHROPIC_MODEL, {"cache_tools": False})
    c.check(
        "cache_tools=False: tools NOT cached",
        "cache_control" not in a.llm._maybe_cache_tools(TOOLS)[-1],
    )

    # --- override=False on Anthropic -> nothing cached ---
    a = make_agent(ANTHROPIC_MODEL, {"override": False})
    sysmsg, last = _system_and_last(a)
    c.check(
        "override=False on Claude: no cache_control anywhere",
        _cache_control_of(sysmsg) is None
        and _cache_control_of(last) is None,
    )

    # --- override=True on OpenAI -> force injection despite auto-provider ---
    a = make_agent(OPENAI_MODEL, {"override": True})
    sysmsg, _ = _system_and_last(a)
    c.check(
        "override=True on OpenAI: forces cache_control injection",
        _cache_control_of(sysmsg) is not None,
    )

    # --- prompt_caching=False -> completely untouched ---
    a = make_agent(ANTHROPIC_MODEL, prompt_caching=False)
    sysmsg, last = _system_and_last(a)
    c.check(
        "prompt_caching=False: no markers at all",
        _cache_control_of(sysmsg) is None
        and _cache_control_of(last) is None,
    )

    # --- OpenAI default -> NOT injected (auto-caches) ---
    a = make_agent(OPENAI_MODEL)
    sysmsg, _ = _system_and_last(a)
    c.check(
        "openai default: system NOT injected (auto-caches)",
        _cache_control_of(sysmsg) is None,
    )

    # --- OpenAI passthrough: prompt_cache_key / prompt_cache_retention ---
    a = make_agent(
        OPENAI_MODEL,
        {
            "prompt_cache_key": "analyst-v1",
            "prompt_cache_retention": "24h",
        },
    )
    cp = {}
    a.llm._apply_cache_request_params(cp)
    c.check(
        "openai: prompt_cache_key passed through",
        cp.get("prompt_cache_key") == "analyst-v1",
    )
    c.check(
        "openai: prompt_cache_retention passed through",
        cp.get("prompt_cache_retention") == "24h",
    )

    print(f"\n  offline: {c.passed} passed, {c.failed} failed\n")
    return c.failed == 0


# ---------------------------------------------------------------------------
# 2. LIVE — run the Agent, confirm real cache activity via usage
# ---------------------------------------------------------------------------
def run_live_check(label, model, cache_config, explicit, env):
    if not _has_key(env):
        print(
            f"  [SKIP] {label:<34} (no credentials: {'/'.join(env)})"
        )
        return None

    try:
        agent = make_agent(model, cache_config)
        # (a) full Agent path with caching on.
        agent.run(TASK)
        # (b) measure on the SAME Agent-built llm.
        agent.llm.return_all = True
        r1 = agent.llm.run(TASK)  # write (or read if smoke warmed it)
        r2 = agent.llm.run(TASK)  # read / hit
    except Exception as e:
        if _is_skippable_error(e):
            print(
                f"  [SKIP] {label:<34} ({type(e).__name__}: {str(e)[:60]})"
            )
            return None
        print(
            f"  [ERROR] {label:<34} {type(e).__name__}: {str(e)[:100]}"
        )
        return False

    m1 = extract_cache_metrics(_as_dict(r1).get("usage"))
    m2 = extract_cache_metrics(_as_dict(r2).get("usage"))

    if explicit:
        ok = m1["created"] > 0 or m2["created"] > 0 or m2["read"] > 0
        detail = f"write={m1['created'] or m2['created']} read(call2)={m2['read']}"
    else:
        ok = m2["cached"] > 0
        detail = f"cached_tokens(call2)={m2['cached']}"

    print(f"  [{'PASS' if ok else 'WARN'}] {label:<34} {detail}")
    if not ok:
        print(
            "         (no cache activity — prefix may be below the provider "
            "minimum, or caching not enabled on this account)"
        )
    return ok


def run_live_checks():
    print("=" * 72)
    print(
        "LIVE — Agent runs against real APIs (providers with credentials)"
    )
    print("=" * 72)

    cases = [
        # Anthropic: full cache_config (1h TTL + tools + both breakpoints).
        (
            "Anthropic full cache_config",
            ANTHROPIC_MODEL,
            {
                "ttl": "1h",
                "cache_system_prompt": True,
                "cache_messages": True,
                "cache_tools": True,
            },
            True,
            ["ANTHROPIC_API_KEY"],
        ),
        # OpenAI: automatic caching + passthrough controls.
        (
            "OpenAI passthrough cache_config",
            OPENAI_MODEL,
            {
                "prompt_cache_key": "analyst-v1",
                "prompt_cache_retention": "24h",
            },
            False,
            ["OPENAI_API_KEY"],
        ),
    ]

    results = [run_live_check(*case) for case in cases]
    ran = [r for r in results if r is not None]
    print()
    if not ran:
        print(
            "  No providers had credentials set — nothing exercised live."
        )
        print(
            "  Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY to test."
        )
    else:
        passed = sum(1 for r in ran if r)
        print(
            f"  Live cases exercised: {len(ran)}  |  cache confirmed: {passed}"
        )
    print()
    return ran


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    offline_only = "--offline" in sys.argv
    offline_ok = run_offline_checks()

    if offline_only:
        print("Offline:", "PASS" if offline_ok else "FAIL")
        sys.exit(0 if offline_ok else 1)

    ran = run_live_checks()
    live_failed = any(r is False for r in ran)
    ok = offline_ok and not live_failed
    print("Overall:", "PASS" if ok else "FAIL / WARN — see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
