"""
Sequential Workflow

from swarms.models import OpenAIChat, Mistral
from swarms.structs import SequentialWorkflow


llm = OpenAIChat(openai_api_key="")
mistral = Mistral()

# Max loops will run over the sequential pipeline twice
workflow = SequentialWorkflow(max_loops=2)

workflow.add("What's the weather in miami", llm)

workflow.add("Create a report on these metrics", mistral)

workflow.run()

"""
from dataclasses import dataclass, field
from typing import List, Any, Dict, Callable, Union
from swarms.models import OpenAIChat
from swarms.structs import Flow


# Define a generic Task that can handle different types of callable objects
@dataclass
class Task:
    description: str
    model: Union[Callable, Flow]
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    result: Any = None

    def execute(self):
        if isinstance(self.model, Flow):
            self.result = self.model.run(*self.args, **self.kwargs)
        else:
            self.result = self.model(*self.args, **self.kwargs)


# SequentialWorkflow class definition using dataclasses
@dataclass
class SequentialWorkflow:
    tasks: List[Task] = field(default_factory=list)
    max_loops: int = 1

    def add(
        self, description: str, model: Union[Callable, Flow], *args, **kwargs
    ) -> None:
        self.tasks.append(
            Task(description=description, model=model, args=list(args), kwargs=kwargs)
        )

    def run(self) -> None:
        for _ in range(self.max_loops):
            for task in self.tasks:
                # Check if the current task can be executed
                if task.result is None:
                    task.execute()
                    # Pass the result as an argument to the next task if it exists
                    next_task_index = self.tasks.index(task) + 1
                    if next_task_index < len(self.tasks):
                        next_task = self.tasks[next_task_index]
                        next_task.args.insert(0, task.result)


# Example usage
api_key = ""  # Your actual API key here

# Initialize the language model
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Initialize the Flow with the language model
flow1 = Flow(llm=llm, max_loops=5, dashboard=True)

# Create another Flow for a different task
flow2 = Flow(llm=llm, max_loops=5, dashboard=True)

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
