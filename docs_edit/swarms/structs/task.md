- This is the class for the Task
- For the constructor, it takes in the description, agent, args, kwargs, result, history, schedule_time, scheduler, trigger, action, condition, priority, and dependencies
- The `execute` method runs the task by calling the agent or model with the arguments and keyword arguments
- It sets a trigger, action, and condition for the task
- Task completion is checked with `is_completed` method
- `add_dependency` adds a task to the list of dependencies
- `set_priority` sets the priority of the task

```python
# Example 1: Creating and executing a Task
from swarms.models import OpenAIChat
from swarms.structs import Agent, Task

agent = Agent(llm=OpenAIChat(openai_api_key=""), max_loops=1, dashboard=False)
task = Task(agent=agent)
task.execute("What's the weather in miami")
print(task.result)

# Example 2: Adding a dependency and setting priority
task2 = Task(description="Task 2", agent=agent)
task.add_dependency(task2)
task.set_priority(1)

# Example 3: Executing a scheduled task
task3 = Task(description="Scheduled Task", agent=agent)
task3.schedule_time = datetime.datetime.now() + datetime.timedelta(minutes=30)
task3.handle_scheduled_task()
print(task3.is_completed())
```
