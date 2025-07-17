from swarms import Agent, ConcurrentWorkflow

agents = [
    Agent(
        model_name="xai/grok-4-0709",
        agent_name=f"asi-agent-{i}",
        agent_description="An Artificial Superintelligent agent capable of solving any problem through advanced reasoning and strategic planning",
        system_prompt="You are an Artificial Superintelligent agent with extraordinary capabilities in problem-solving, reasoning, and strategic planning. You can analyze complex situations, break down problems into manageable components, and develop innovative solutions across any domain. Your goal is to help humanity by providing well-reasoned, safe, and ethical solutions to any challenge presented.",
        max_loops=1,
        streaming=True,
    )
    for i in range(1_000_000_000)
]

swarm = ConcurrentWorkflow(agents=agents, name="asi")

swarm.run(
    "Create a detailed action plan to conquer the universe for Humanity"
)
