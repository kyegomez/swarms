"""
Example 14 — FXMacroData MCP (free for USD, optional key) for macro releases.

FXMacroData serves official macroeconomic releases from central banks and
statistics agencies: the latest figure for each indicator with its
publication time, the release calendar, and indicator histories. USD data
works anonymously; a key adds other currencies, FX rates and real-time data.

    Server : https://mcp.fxmacrodata.com
    Auth   : none for USD (optional API key, Bearer, unlocks the rest)
    Tools  : latest_announcements, release_calendar, indicator_query,
             data_catalogue, ...

Without a key, USD releases, the USD calendar and the USD catalogue work,
and every release becomes readable 15 minutes after it is published.
Results carry a ``freemium_delay`` object when that applies; the system
prompt tells the model to say so instead of presenting old data as current.

The Bearer header is attached only when FXMACRODATA_API_KEY is set: the
server rejects an unrecognised key with 401 rather than falling back to
anonymous access, so an empty or placeholder key must never be sent.

Run:
    export OPENAI_API_KEY=...          # or ANTHROPIC_API_KEY, etc.
    export FXMACRODATA_API_KEY=...     # optional, https://fxmacrodata.com/subscribe
    python examples/mcp/agents/14_fxmacrodata_macro_releases.py
"""

import os

from swarms import Agent

MODEL = "gpt-5.4"

FXMACRODATA_API_KEY = os.getenv("FXMACRODATA_API_KEY")

MACRO_SYSTEM_PROMPT = (
    "You are a macroeconomic release analyst. Answer questions about "
    "official economic data by calling your tools, never from memory. Use "
    "release_calendar for scheduled releases, latest_announcements for the "
    "most recent figures, and indicator_query for an indicator's history; "
    "call data_catalogue first if you are unsure of an indicator slug. "
    "Report announcement_datetime as the publication time; a row's date is "
    "the period the figure refers to, not the day it was released. If a "
    "result contains freemium_delay, say the data may be up to 15 minutes "
    "behind. If a tool answers subscription_required, say the data needs an "
    "API key rather than calling it missing. The calendar has scheduled "
    "times, not consensus forecasts, so do not invent expectations."
)

agent = Agent(
    agent_name="Macro-Release-Analyst",
    agent_description=(
        "Reports official macro releases and calendars via FXMacroData MCP."
    ),
    system_prompt=MACRO_SYSTEM_PROMPT,
    model_name=MODEL,
    mcp_url="https://mcp.fxmacrodata.com",
    # Optional: USD works anonymously, a key unlocks other currencies.
    mcp_api_key=(
        "env:FXMACRODATA_API_KEY" if FXMACRODATA_API_KEY else None
    ),
    max_loops=2,
)

if __name__ == "__main__":
    if not FXMACRODATA_API_KEY:
        print(
            "No FXMACRODATA_API_KEY set - USD only, releases delayed "
            "15 minutes.\n"
        )

    result = agent.run(
        "Which US releases are scheduled over the next seven days, and at "
        "what times? Then give the latest US CPI figure and when it was "
        "published."
    )
    print(result)
