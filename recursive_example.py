import os
from dotenv import load_dotenv
from swarms import OpenAIChat, Task, RecursiveWorkflow, Agent

# Load environment variables from .env file
load_dotenv()

# Load environment variables
llm = OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = Agent(llm=llm, max_loops=1)

# Create a workflow
workflow = RecursiveWorkflow(stop_token="<DONE>")

# Create tasks
task1 = Task(agent, "What's the weather in miami")
task2 = Task(agent, "What's the weather in new york")
task3 = Task(agent, "What's the weather in london")

# Add tasks to the workflow
workflow.add(task1)
workflow.add(task2)
workflow.add(task3)

# Run the workflow
workflow.run()
