from swarms import Agent, MultiAgentRouter

# Example usage:
if __name__ == "__main__":
    # Define some example agents
    agents = [
        Agent(
            agent_name="ResearchAgent",
            description="Specializes in researching topics and providing detailed, factual information",
            system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
            model_name="openai/gpt-4o",
        ),
        Agent(
            agent_name="CodeExpertAgent",
            description="Expert in writing, reviewing, and explaining code across multiple programming languages",
            system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
            model_name="openai/gpt-4o",
        ),
        Agent(
            agent_name="WritingAgent",
            description="Skilled in creative and technical writing, content creation, and editing",
            system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
            model_name="openai/gpt-4o",
        ),
    ]

    # Initialize routers with different configurations
    router_execute = MultiAgentRouter(
        agents=agents, execute_task=True
    )

    # Example task
    task = "Write a Python function to calculate fibonacci numbers"

    try:
        # Process the task with execution
        print("\nWith task execution:")
        result_execute = router_execute.route_task(task)
        print(result_execute)

    except Exception as e:
        print(f"Error occurred: {e!s}")
