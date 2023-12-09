from swarms.models import OpenAIChat
from swarms.structs import Flow
from swarms.structs.sequential_workflow import SequentialWorkflow

# Example usage
llm = OpenAIChat(
    temperature=0.5,
    max_tokens=3000,
)

# Initialize the Flow with the language flow
flow1 = Flow(llm=llm, max_loops=1, dashboard=False)

# Create another Flow for a different task
flow2 = Flow(llm=llm, max_loops=1, dashboard=False)

# Create the workflow
workflow = SequentialWorkflow(max_loops=1)

# Add tasks to the workflow
workflow.add("Generate a 10,000 word blog on health and wellness.", flow1)

# Suppose the next task takes the output of the first task as input
workflow.add("Summarize the generated blog", flow2)

# Run the workflow
workflow.run()

# Output the results
for task in workflow.tasks:
    print(f"Task: {task.description}, Result: {task.result}")
