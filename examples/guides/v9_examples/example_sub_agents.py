import os

from swarms.structs.agent import Agent

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Create the main agent with autonomous mode
main_agent = Agent(
    agent_name="Main Research Coordinator",
    agent_description="A coordinator agent that delegates research tasks to specialized sub-agents",
    model_name="gpt-4.1",
    max_loops="auto",  # Enable autonomous mode
)

# Example task that will trigger sub-agent creation and task delegation
task = """
Research the following three topics and provide a comprehensive analysis:
1. The impact of artificial intelligence on healthcare
2. Recent advances in quantum computing
3. Climate change mitigation strategies

For each topic, I need you to:
- Create a specialized sub-agent that can focus on that specific domain
- Assign the research task to each sub-agent
- Compile all results into a final comprehensive report
"""

if __name__ == "__main__":
    result = main_agent.run(task)
    print(result)
