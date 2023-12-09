from swarms.models import OpenAIChat, BioGPT, Anthropic
from swarms.structs import Flow
from swarms.structs.sequential_workflow import SequentialWorkflow


# Example usage
api_key = ""  # Your actual API key here

# Initialize the language flow
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

biochat = BioGPT()

# Use Anthropic
anthropic = Anthropic()

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
