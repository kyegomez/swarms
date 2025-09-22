from swarms.structs.agent import Agent
from swarms.structs.multi_agent_router import MultiAgentRouter

# Example usage:
agents = [
    Agent(
        agent_name="ResearchAgent",
        agent_description="Specializes in researching topics and providing detailed, factual information",
        system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
    ),
    Agent(
        agent_name="CodeExpertAgent",
        agent_description="Expert in writing, reviewing, and explaining code across multiple programming languages",
        system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
    ),
    Agent(
        agent_name="WritingAgent",
        agent_description="Skilled in creative and technical writing, content creation, and editing",
        system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
    ),
]

# Initialize routers with different configurations
router_execute = MultiAgentRouter(
    agents=agents, temperature=0.5, model="claude-sonnet-4-20250514"
)

# Example task: Remake the Fibonacci task
task = "Use all the agents available to you to remake the Fibonacci function in Python, providing both an explanation and code."
result_execute = router_execute.run(task)
print(result_execute)
