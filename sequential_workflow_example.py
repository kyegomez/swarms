from swarms.models import OpenAIChat, BioGPT, Anthropic
from swarms.structs import Flow
from swarms.structs.sequential_workflow import SequentialWorkflow
import os

# Example usage

openai_api_key = os.environ.get("OPENAI_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")


# Initialize the language flow
llm = OpenAIChat(
    openai_api_key=openai_api_key,
    temperature=0.5,
    max_tokens=3000,
)

biochat = BioGPT()

# Use Anthropic
anthropic = Anthropic(anthropic_api_key=anthropic_api_key)

# Initialize the agent with the language flow
agent1 = Flow(llm=llm, max_loops=1, dashboard=False)

# Create another agent for a different task
agent2 = Flow(llm=llm, max_loops=1, dashboard=False)

# Create another agent for a different task
agent3 = Flow(llm=biochat, max_loops=1, dashboard=False)

# agent4 = Flow(llm=anthropic, max_loops="auto")

# Create the workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add("Generate a 10,000 word blog on health and wellness.", agent1)

# Suppose the next task takes the output of the first task as input
workflow.add("Summarize the generated blog", agent2)

workflow.add("Create a references sheet of materials for the curriculm", agent3)

# Run the workflow
workflow.run()

# Output the results
for task in workflow.tasks:
    print(f"Task: {task.description}, Result: {task.result}")
