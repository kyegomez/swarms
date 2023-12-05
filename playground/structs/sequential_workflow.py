from swarms.models import OpenAIChat
from swarms.structs import Agent
from swarms.structs.sequential_workflow import SequentialWorkflow

# Example usage
llm = OpenAIChat(
    temperature=0.5,
    max_tokens=3000,
)

# Initialize the Agent with the language agent
flow1 = Agent(llm=llm, max_loops=1, dashboard=False)

# Create another Agent for a different task
flow2 = Agent(llm=llm, max_loops=1, dashboard=False)

# Create the workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add(
    "Generate a 10,000 word blog on health and wellness.", flow1
)

# Suppose the next task takes the output of the first task as input
workflow.add("Summarize the generated blog", flow2)

# Run the workflow
workflow.run()

# Output the results
for task in workflow.tasks:
    print(f"Task: {task.description}, Result: {task.result}")
