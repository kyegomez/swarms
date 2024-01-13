import os
from dotenv import load_dotenv
from swarms import OpenAIChat, Task, ConcurrentWorkflow, Agent

# Load environment variables from .env file
load_dotenv()

# Load environment variables
llm = OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = Agent(
    system_prompt=None,
    llm=llm,
    max_loops=1,
)

# Create a workflow
workflow = ConcurrentWorkflow(max_workers=3)

# Create tasks
task1 = Task(agent=agent, description="What's the weather in miami")
task2 = Task(
    agent=agent, description="What's the weather in new york"
)
task3 = Task(agent=agent, description="What's the weather in london")

# Add tasks to the workflow
workflow.add(tasks=[task1, task2, task3])

# Run the workflow and print each task result
workflow.run()
