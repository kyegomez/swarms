import json

from swarms import Agent, SwarmRouter, RESPOND_TOOL

# Build three opinionated agents for the discussion.
researcher = Agent(
    agent_name="Researcher",
    system_prompt=(
        "You are a research-minded agent who values evidence and citations. "
        "Speak only when you can add a specific fact, study, or measurement."
    ),
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
    output_type="final",
)

skeptic = Agent(
    agent_name="Skeptic",
    system_prompt=(
        "You push back on weak claims and ask sharp questions. "
        "Speak only when you spot an unsupported assumption or logical gap."
    ),
    model_name="claude-sonnet-4-6",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
    output_type="final",
)

builder = Agent(
    agent_name="Builder",
    system_prompt=(
        "You turn ideas into concrete next steps. "
        "Speak only when you can name a specific action, tradeoff, or design choice."
    ),
    model_name="gpt-5.5",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
    output_type="final",
)

agents = [researcher, skeptic, builder]

router = SwarmRouter(
    name="design-debate",
    description="A self-selecting groupchat that debates architecture decisions.",
    agents=agents,
    swarm_type="GroupChat",
    max_loops=6,  # hard cap on total messages posted
    output_type="dict",  # how SwarmRouter formats the final return
)

result = router.run(
    "Should we back our agent memory layer with a vector database or a knowledge graph? "
    "Discuss the tradeoffs and converge on a recommendation."
)

print(json.dumps(result, indent=2, default=str))
