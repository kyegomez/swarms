from swarms import Agent, MCPDeployer

# The simplest deployment: one agent, one static key.
researcher = Agent(
    agent_name="Researcher",
    agent_description="Answers research questions with a short, sourced summary.",
    system_prompt="You are a careful researcher. Answer in one short paragraph.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

deployer = MCPDeployer(
    researcher,
    api_keys=["sk-local-dev"],
    port=8000,
)

if __name__ == "__main__":
    deployer.run()
