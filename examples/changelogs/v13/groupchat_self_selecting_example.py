"""Async self-selecting GroupChat (v12.0.3).

No rounds or speaker-selection functions: every agent listens in parallel
and decides on its own whether to chime in via a forced respond(score,
message) call. Replies scoring above `threshold` are broadcast; the chat
ends after `max_loops` messages or `idle_timeout` seconds of silence.
The RESPOND_TOOL is auto-injected into every agent since 12.0.3.
"""

from swarms import Agent, GroupChat

optimist = Agent(
    agent_name="Optimist",
    system_prompt="Argue the upside.",
    model_name="gpt-4.1",
    max_loops=1,
)
pessimist = Agent(
    agent_name="Pessimist",
    system_prompt="Argue the risks.",
    model_name="gpt-4.1",
    max_loops=1,
)

chat = GroupChat(
    agents=[
        optimist,
        pessimist,
    ],  # RESPOND_TOOL is auto-injected since 12.0.3
    max_loops=10,  # hard cap on total messages
    threshold=0.5,  # min desire-to-speak score (0..1) to publish
    idle_timeout=8.0,  # seconds of silence before the chat ends
)

result = chat.run("Should we adopt AI for medical diagnosis?")
print(result)
