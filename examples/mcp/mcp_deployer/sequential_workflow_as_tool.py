from swarms import Agent, MCPDeployer, SequentialWorkflow

researcher = Agent(
    agent_name="Researcher",
    system_prompt="Research the topic and list the five most important facts.",
    model_name="claude-sonnet-4-6",
    max_loops=1,
    print_on=False,
)
writer = Agent(
    agent_name="Writer",
    system_prompt="Turn the facts into a tight three-paragraph briefing.",
    model_name="claude-sonnet-4-6",
    max_loops=1,
    print_on=False,
)

pipeline = SequentialWorkflow(
    name="Briefing-Pipeline",
    description="Researches a topic and writes a three-paragraph briefing.",
    agents=[researcher, writer],
    max_loops=1,
)

deployer = MCPDeployer(
    pipeline,
    tool_name="write_briefing",
    api_keys=["sk-local-dev"],
    port=8001,
    timeout=300,  # two model calls in series; give it room
)

if __name__ == "__main__":
    deployer.run()
