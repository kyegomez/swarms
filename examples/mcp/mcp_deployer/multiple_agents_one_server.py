from swarms import Agent, MCPDeployer, SequentialWorkflow

researcher = Agent(
    agent_name="Researcher",
    agent_description="Finds the key facts on a topic.",
    system_prompt="List the five most important facts about the topic.",
    model_name="claude-sonnet-4-6",
    max_loops=1,
    print_on=False,
)

critic = Agent(
    agent_name="Critic",
    agent_description="Finds the weakest point in an argument.",
    system_prompt="Name the single weakest point in the text and why.",
    model_name="openrouter/moonshotai/kimi-k3",
    max_loops=1,
    print_on=False,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="Turn the facts into a tight three-paragraph briefing.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

briefing = SequentialWorkflow(
    name="Briefing-Pipeline",
    description="Researches a topic, then writes a briefing from the facts.",
    agents=[researcher, writer],
    max_loops=1,
)


def word_count(task: str) -> int:
    """Count the words in the text."""
    return len(task.split())


deployer = MCPDeployer(
    {
        "research": researcher,
        "critique": critic,
        "write_briefing": briefing,
        "word_count": word_count,
    },
    name="Editorial-Team",
    api_keys=["sk-local-dev"],
    port=8006,
    timeout=300,
)

deployer.add_tool(lambda task: task[::-1], name="reverse")

if __name__ == "__main__":
    deployer.run()
