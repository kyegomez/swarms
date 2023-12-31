import os 
from dotenv import load_dotenv 
from swarms.models import OpenAIChat, Task, ConcurrentWorkflow

# Load environment variables from .env file
load_dotenv()

# Load environment variables
llm = OpenAIChat(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Create a workflow
workflow = ConcurrentWorkflow(max_workers=5)

# Create tasks
task1 = Task(llm, "What's the weather in miami")
task2 = Task(llm, "What's the weather in new york")
task3 = Task(llm, "What's the weather in london")

# Add tasks to the workflow
workflow.add(task1)
workflow.add(task2)
workflow.add(task3)

# Run the workflow
workflow.run()
