from swarms import Agent, MCPDeployer

assistant = Agent(
    agent_name="Desktop-Assistant",
    agent_description="A general assistant exposed to a desktop MCP host.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

deployer = MCPDeployer(
    assistant,
    transport="stdio",
    allow_anonymous=True,
)

if __name__ == "__main__":
    deployer.run()
