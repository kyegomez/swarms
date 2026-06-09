import json

from swarms import RESPOND_TOOL, Agent, GroupChat

a1 = Agent(
    agent_name="Researcher",
    system_prompt="You are a research-minded agent who values evidence.",
    model_name="gpt-5.5",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
    output_type="final",
)

a2 = Agent(
    agent_name="Skeptic",
    system_prompt="You push back on weak claims and ask sharp questions.",
    model_name="claude-haiku-4-5",
    max_loops=1,
    tools_list_dictionary=[RESPOND_TOOL],
    persistent_memory=False,
    output_type="final",
)

a3 = Agent(
    agent_name="Builder",
    system_prompt="You turn ideas into concrete next steps.",
    model_name="gpt-5.5",
    max_loops=1,
    tools_list_dictionary=[RESPOND_TOOL],
    persistent_memory=False,
    output_type="final",
)

chat = GroupChat(
    agents=[a1, a2, a3],
    max_loops=4,
    threshold=0.6,
    output_type="dict",
)
result = chat.run(
    "Should we use vector databases or knowledge graphs for agent memory?"
)
print(json.dumps(result, indent=4))
