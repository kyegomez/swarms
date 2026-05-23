"""
Stream extended-thinking tokens in real time via structured events.

Covers three modes:
  1. streaming_on=True  + streaming_callback (dict-typed) + streaming_events=True
  2. run_stream(with_events=True)
  3. stream=True (detailed mode) + streaming_events=True

Requires ANTHROPIC_API_KEY. Uses claude-sonnet-4-6 with extended thinking.
"""

import sys

from swarms import Agent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "claude-sonnet-4-6"
TASK = (
    "Think step by step before answering. A chicken and a half lays "
    "an egg and a half in a day and a half. How many eggs does one "
    "chicken lay per day? Show your reasoning, then give the final answer."
)


def on_event(event: dict) -> None:
    evt_type = event.get("type")
    if evt_type == "thinking":
        print(event["token"], end="", flush=True)
    elif evt_type == "content":
        print(event["token"], end="", flush=True)
    elif evt_type in (
        "thinking_start",
        "thinking_end",
        "content_start",
        "content_end",
    ):
        print(f"\n[{evt_type}]", flush=True)


def main() -> None:
    agent = Agent(
        agent_name="Reasoner",
        model_name=MODEL,
        thinking_tokens=2000,
        streaming_on=True,
        streaming_callback=on_event,
        streaming_events=True,
        print_on=True,
        max_loops=1,
        verbose=False,
    )

    print("=== run() with streaming_events=True ===\n")
    print("[thinking stream]")
    agent.run(TASK)

    print("\n\n=== run_stream(with_events=True) ===\n")
    print("[thinking stream]")
    saw_thinking = False
    for evt in agent.run_stream(TASK, with_events=True):
        if evt.get("type") == "thinking":
            saw_thinking = True
            print(evt["token"], end="", flush=True)
        elif evt.get("type") == "content":
            print(evt["token"], end="", flush=True)
    print()
    if not saw_thinking:
        print(
            "WARNING: no thinking events received — check thinking_tokens "
            "and model support."
        )

    print("\n\n=== stream=True (detailed mode) + streaming_events=True ===\n")
    detailed_agent = Agent(
        agent_name="Reasoner-Detailed",
        model_name=MODEL,
        thinking_tokens=2000,
        stream=True,
        streaming_callback=on_event,
        streaming_events=True,
        print_on=False,
        max_loops=1,
        verbose=False,
    )
    saw_thinking = False
    saw_content = False

    original_on_event = on_event.__code__

    def on_event_detailed(event: dict) -> None:
        global saw_thinking, saw_content
        evt_type = event.get("type")
        if evt_type == "thinking":
            saw_thinking = True
            print(event["token"], end="", flush=True)
        elif evt_type == "content":
            saw_content = True
            print(event["token"], end="", flush=True)
        elif evt_type in ("thinking_start", "thinking_end", "content_start", "content_end"):
            print(f"\n[{evt_type}]", flush=True)

    saw_thinking = False
    saw_content = False
    detailed_agent.streaming_callback = on_event_detailed
    detailed_agent.run(TASK)
    print()
    if not saw_thinking:
        print("WARNING: no thinking events in detailed mode.")
    if not saw_content:
        print("WARNING: no content events in detailed mode.")
    if saw_thinking and saw_content:
        print("\nPASS: both thinking and content events have 'type' field in detailed mode.")


if __name__ == "__main__":
    main()
