from dataclass import dataclass, field
from swarms.structs.agent import Agent
from typing import Optional
from typing import List, Dict, Any, Sequence


@dataclass
class Task:
    """
    Task is a unit of work that can be executed by a set of agents.

    A task is defined by a task name and a set of agents that can execute the task.
    The task can also have a set of dependencies, which are the names of other tasks
    that must be executed before this task can be executed.

    Args:
        id (str): The name of the task.
        description (Optional[str]): A description of the task.
        task (str): The name of the task.
        result (Any): The result of the task.
        agents (Sequence[Agent]): A list of agents that can execute the task.
        dependencies (List[str], optional): A list of task names that must be executed before this task can be executed. Defaults to [].
        args (List[Any], optional): A list of arguments to pass to the agents. Defaults to field(default_factory=list).
        kwargs (List[Any], optional): A list of keyword arguments to pass to the agents. Defaults to field(default_factory=list).

    Methods:
        execute: Executes the task by passing the results of the parent tasks to the agents.

    Examples:
    import os
    from swarms.models import OpenAIChat
    from swarms.structs import Agent
    from swarms.structs.sequential_workflow import SequentialWorkflow
    from dotenv import load_dotenv

    load_dotenv()

    # Load the environment variables
    api_key = os.getenv("OPENAI_API_KEY")


    # Initialize the language agent
    llm = OpenAIChat(
        openai_api_key=api_key,
        temperature=0.5,
        max_tokens=3000,
    )


    # Initialize the agent with the language agent
    agent1 = Agent(llm=llm, max_loops=1)

    # Create another agent for a different task
    agent2 = Agent(llm=llm, max_loops=1)

    # Create the workflow
    workflow = SequentialWorkflow(max_loops=1)

    # Add tasks to the workflow
    workflow.add(
        agent1, "Generate a 10,000 word blog on health and wellness.",
    )

    # Suppose the next task takes the output of the first task as input
    workflow.add(
        agent2, "Summarize the generated blog",
    )

    # Run the workflow
    workflow.run()

    # Output the results
    for task in workflow.tasks:
        print(f"Task: {task.description}, Result: {task.result}")

    """

    def __init__(
        self,
        id: str,
        description: Optional[str],
        task: str,
        result: Any,
        agents: Sequence[Agent],
        dependencies: List[str] = [],
        args: List[Any] = field(default_factory=list),
        kwargs: List[Any] = field(default_factory=list),
    ):
        self.id = id
        self.description = description
        self.task = task
        self.result = result
        self.agents = agents
        self.dependencies = dependencies
        self.results = []
        self.args = args
        self.kwargs = kwargs

    def execute(self, parent_results: Dict[str, Any]):
        """Executes the task by passing the results of the parent tasks to the agents.

        Args:
            parent_results (Dict[str, Any]): A dictionary of task names and their results.

        Examples:
        """
        args = [parent_results[dep] for dep in self.dependencies]
        for agent in self.agents:
            if isinstance(agent, Agent):
                if "prompt" in self.kwargs:
                    self.kwargs["prompt"] += (
                        f"\n\nPrevious output: {self.results[-1]}"
                        if self.results
                        else ""
                    )
                else:
                    self.kwargs["prompt"] = (
                        f"Main task: {self.description}"
                        + (
                            f"\n\nPrevious output: {self.results[-1]}"
                            if self.results
                            else ""
                        )
                    )
                result = agent.run(
                    self.description, *args, **self.kwargs
                )
            else:
                result = agent(self.description, *args, **self.kwargs)
            self.results.append(result)
            args = [result]
            self.history.append(result)
