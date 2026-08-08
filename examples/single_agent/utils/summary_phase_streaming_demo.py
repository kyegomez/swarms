from dotenv import load_dotenv
from swarms import Agent

load_dotenv()

tokens = []

agent = Agent(
    model_name="anthropic/claude-sonnet-4-20250514",
    temperature=1.0,
    max_loops="auto",
    stream=True,
    print_on=False,
    retry_attempts=8,
    selected_tools=[
        "create_plan",
        "think",
        "subtask_done",
        "complete_task",
    ],
)

try:
    result = agent.run(
        "Write exactly one short sentence: Streaming callback verified.",
        streaming_callback=lambda t: tokens.append(
            t.get("token") if isinstance(t, dict) else t
        ),
    )
    ran_ok = True
except Exception:
    result, ran_ok = "", False

stream_tokens = [t for t in tokens if t]
print("=== SUMMARY_STREAMING ===")
print("STREAM_TOKEN_COUNT:", len(stream_tokens))
print("SUMMARY_STREAMING_SUCCESS:", "YES" if stream_tokens else "NO")
print(
    "STREAM_PREVIEW:",
    "".join(str(t) for t in stream_tokens[:8])[:180],
)
print("=== RUN_OUTPUT ===")
print(
    str(result)[:300]
    if ran_ok
    else "Execution ended before final response; streaming callback signal is shown above."
)
