from swarms import OpenAIFunctionCaller
from swarms.structs.hiearchical_swarm import (
    HierarchicalAgentSwarm,
    SwarmSpec,
    HIEARCHICAL_AGENT_SYSTEM_PROMPT,
)

director = (
    OpenAIFunctionCaller(
        system_prompt=HIEARCHICAL_AGENT_SYSTEM_PROMPT,
        max_tokens=3000,
        temperature=0.4,
        base_model=SwarmSpec,
        parallel_tool_calls=False,
    ),
)

# Initialize the hierarchical agent swarm with the necessary parameters
swarm = HierarchicalAgentSwarm(
    name="Hierarchical Swarm Example",
    description="A swarm of agents to promote the swarms workshop",
    director=director,
    max_loops=1,
    create_agents_on=True,
)

# Run the swarm with a task
agents = swarm.run(
    """
    Create a swarm of agents for a marketing campaign to promote
    the swarms workshop: [Workshop][Automating Business Operations with Hierarchical Agent Swarms][Swarms Framework + GPT4o],
    create agents for twitter, linkedin, and emails, facebook, instagram.

    The date is Saturday, August 17 4:00 PM - 5:00 PM

    Link is: https://lu.ma/ew4r4s3i


    """
)
