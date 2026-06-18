"""Smoke test for the dynamic GroupChat."""

from swarms import Agent
from swarms.structs.groupchat import GroupChat, RESPOND_TOOL

afu = Agent(
    agent_name="afu",
    system_prompt="You are afu, a creative copywriter for Xiaohongshu beauty posts.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

nunu = Agent(
    agent_name="nunu",
    system_prompt="You are nunu, a beauty product reviewer who critiques copy for authenticity.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

chat = GroupChat(
    agents=[afu, nunu],
    max_loops=8,
    threshold=0.5,
    idle_timeout=8.0,
)

response = chat.run(
    "@afu Write a Xiaohongshu lipstick advertisement copy"
)
print(response)
