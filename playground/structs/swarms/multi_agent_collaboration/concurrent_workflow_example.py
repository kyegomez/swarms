import os

from dotenv import load_dotenv

from swarms import Agent, ConcurrentWorkflow, OpenAIChat, Task

# Load environment variables from .env file
load_dotenv()

# Load environment variables
llm = OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = Agent(llm=llm, max_loops=1)

# Create a workflow
workflow = ConcurrentWorkflow(max_workers=5)

task = (
    "Generate a report on how small businesses spend money and how"
    " can they cut 40 percent of their costs"
)

# Create tasks
task1 = Task(agent, task)
task2 = Task(agent, task)
task3 = Task(agent, task)

# Add tasks to the workflow
workflow.add(tasks=[task1, task2, task3])

# Run the workflow
workflow.run()
