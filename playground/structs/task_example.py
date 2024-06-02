import os
from dotenv import load_dotenv
from swarms.structs import Agent, OpenAIChat, Task

# Load the environment variables
load_dotenv()


# Define a function to be used as the action
def my_action():
    print("Action executed")


# Define a function to be used as the condition
def my_condition():
    print("Condition checked")
    return True


# Create an agent
agent = Agent(
    llm=OpenAIChat(openai_api_key=os.environ["OPENAI_API_KEY"]),
    max_loops=1,
    dashboard=False,
)

# Create a task
task = Task(
    description=(
        "Generate a report on the top 3 biggest expenses for small"
        " businesses and how businesses can save 20%"
    ),
    agent=agent,
)

# Set the action and condition
task.set_action(my_action)
task.set_condition(my_condition)

# Execute the task
print("Executing task...")
task.run()

# Check if the task is completed
if task.is_completed():
    print("Task completed")
else:
    print("Task not completed")

# Output the result of the task
print(f"Task result: {task.result}")
