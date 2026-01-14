from swarms.structs.agent import Agent


def main():
    # Create specialized agents
    research_agent = Agent(
        agent_name="ResearchAgent",
        agent_description="Specializes in researching topics and providing detailed, factual information",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
    )

    code_agent = Agent(
        agent_name="CodeExpertAgent",
        agent_description="Expert in writing, reviewing, and explaining code across multiple programming languages",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
    )

    writing_agent = Agent(
        agent_name="WritingAgent",
        agent_description="Skilled in creative and technical writing, content creation, and editing",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
    )

    # Create a coordinator agent with handoffs enabled
    coordinator = Agent(
        agent_name="CoordinatorAgent",
        agent_description="Coordinates tasks and delegates to specialized agents",
        model_name="gpt-4o-mini",
        max_loops=1,
        handoffs=[research_agent, code_agent, writing_agent],
        system_prompt="You are a coordinator agent. Analyze tasks and delegate them to the most appropriate specialized agent using the handoff_task tool. You can delegate to multiple agents if needed.",
        output_type="all",
    )

    # Test 2: Complex task that might need multiple agents
    task2 = "Call all the agents available to you and ask them how they are doing"
    result2 = coordinator.run(task=task2)
    print(result2)


if __name__ == "__main__":
    results = main()
