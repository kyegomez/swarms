from swarms import Agent, OpenAIChat, SequentialWorkflow, Task

# Example usage
llm = OpenAIChat(
    temperature=0.5,
    max_tokens=3000,
)

# Initialize the Agent with the language agent
agent1 = Agent(
    agent_name="John the writer",
    llm=llm,
    max_loops=0,
    dashboard=False,
)
task1 = Task(
    agent=agent1,
    description="Write a 1000 word blog about the future of AI",
)

# Create another Agent for a different task
agent2 = Agent("Summarizer", llm=llm, max_loops=1, dashboard=False)
task2 = Task(
    agent=agent2,
    description="Summarize the generated blog",
)

# Create the workflow
workflow = SequentialWorkflow(
    name="Blog Generation Workflow",
    description=(
        "A workflow to generate and summarize a blog about the future"
        " of AI"
    ),
    max_loops=1,
    autosave=True,
    dashboard=False,
)

# Add tasks to the workflow
workflow.add(tasks=[task1, task2])

# Run the workflow
workflow.run()

# # Output the results
for task in workflow.tasks:
    print(f"Task: {task.description}, Result: {task.result}")
