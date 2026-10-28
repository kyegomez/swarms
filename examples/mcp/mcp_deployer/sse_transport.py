from swarms import Agent, MCPDeployer

translator = Agent(
    agent_name="Translator",
    agent_description="Translates text into French.",
    system_prompt="Translate the text into French. Return only the translation.",
    model_name="openrouter/z-ai/glm-5.3",
    max_loops=1,
    print_on=False,
)

deployer = MCPDeployer(
    translator,
    transport="sse",
    api_keys=["sk-local-dev"],
    port=8005,
)

if __name__ == "__main__":
    deployer.run()
