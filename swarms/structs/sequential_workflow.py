import concurrent.futures
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import logging
from termcolor import colored

from swarms.structs.agent import Agent


# Define a generic Task that can handle different types of callable objects
@dataclass
class Task:
    """
    Task class for running a task in a sequential workflow.


    Args:
        description (str): The description of the task.
        agent (Union[Callable, Agent]): The model or agent to execute the task.
        args (List[Any]): Additional arguments to pass to the task execution.
        kwargs (Dict[str, Any]): Additional keyword arguments to pass to the task execution.
        result (Any): The result of the task execution.
        history (List[Any]): The history of the task execution.

    Methods:
        execute: Execute the task.


    Examples:
    >>> from swarms.structs import Task, Agent
    >>> from swarms.models import OpenAIChat
    >>> agent = Agent(llm=OpenAIChat(openai_api_key=""), max_loops=1, dashboard=False)
    >>> task = Task(description="What's the weather in miami", agent=agent)
    >>> task.execute()
    >>> task.result

    """

    description: str
    agent: Union[Callable, Agent]
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    history: List[Any] = field(default_factory=list)

    def execute(self):
        """
        Execute the task.

        Raises:
            ValueError: If a Agent instance is used as a task and the 'task' argument is not provided.
        """
        if isinstance(self.agent, Agent):
            # Add a prompt to notify the Agent of the sequential workflow
            if "prompt" in self.kwargs:
                self.kwargs["prompt"] += (
                    f"\n\nPrevious output: {self.result}"
                    if self.result
                    else ""
                )
            else:
                self.kwargs["prompt"] = (
                    f"Main task: {self.description}"
                    + (
                        f"\n\nPrevious output: {self.result}"
                        if self.result
                        else ""
                    )
                )
            self.result = self.agent.run(*self.args, **self.kwargs)
        else:
            self.result = self.agent(*self.args, **self.kwargs)

        self.history.append(self.result)


# SequentialWorkflow class definition using dataclasses
@dataclass
class SequentialWorkflow:
    """
    SequentialWorkflow class for running a sequence of tasks using N number of autonomous agents.

    Args:
        max_loops (int): The maximum number of times to run the workflow.
        dashboard (bool): Whether to display the dashboard for the workflow.


    Attributes:
        tasks (List[Task]): The list of tasks to execute.
        max_loops (int): The maximum number of times to run the workflow.
        dashboard (bool): Whether to display the dashboard for the workflow.


    Examples:
    >>> from swarms.models import OpenAIChat
    >>> from swarms.structs import SequentialWorkflow
    >>> llm = OpenAIChat(openai_api_key="")
    >>> workflow = SequentialWorkflow(max_loops=1)
    >>> workflow.add("What's the weather in miami", llm)
    >>> workflow.add("Create a report on these metrics", llm)
    >>> workflow.run()
    >>> workflow.tasks

    """

    name: str = None
    description: str = None
    tasks: List[Task] = field(default_factory=list)
    max_loops: int = 1
    autosave: bool = False
    saved_state_filepath: Optional[str] = (
        "sequential_workflow_state.json"
    )
    restore_state_filepath: Optional[str] = None
    dashboard: bool = False

    def add(
        self,
        agent: Union[Callable, Agent],
        task: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Add a task to the workflow.

        Args:
            agent (Union[Callable, Agent]): The model or agent to execute the task.
            task (str): The task description or the initial input for the Agent.

            *args: Additional arguments to pass to the task execution.
            **kwargs: Additional keyword arguments to pass to the task execution.
        """
        try:
            # If the agent is a Agent instance, we include the task in kwargs for Agent.run()
            if isinstance(agent, Agent):
                kwargs["task"] = (
                    task  # Set the task as a keyword argument for Agent
                )

            # Append the task to the tasks list
            self.tasks.append(
                Task(
                    description=task,
                    agent=agent,
                    args=list(args),
                    kwargs=kwargs,
                )
            )
        except Exception as error:
            print(
                colored(
                    f"Error adding task to workflow: {error}", "red"
                ),
            )

    def reset_workflow(self) -> None:
        """Resets the workflow by clearing the results of each task."""
        try:
            
            for task in self.tasks:
                task.result = None
        except Exception as error:
            print(
                colored(
                    f"Error resetting workflow: {error}", "red"
                ),
            )

    def get_task_results(self) -> Dict[str, Any]:
        """
        Returns the results of each task in the workflow.

        Returns:
            Dict[str, Any]: The results of each task in the workflow
        """
        try:
            return {
                task.description: task.result for task in self.tasks
            }
        except Exception as error:
            print(
                colored(
                    f"Error getting task results: {error}", "red"
                ),
            )

    def remove_task(self, task: str) -> None:
        """Remove tasks from sequential workflow"""
        try:    
            self.tasks = [
                task for task in self.tasks if task.description != task
            ]
        except Exception as error:
            print(
                colored(
                    f"Error removing task from workflow: {error}", "red"
                ),
            )

    def update_task(self, task: str, **updates) -> None:
        """
        Updates the arguments of a task in the workflow.

        Args:
            task (str): The description of the task to update.
            **updates: The updates to apply to the task.

        Raises:
            ValueError: If the task is not found in the workflow.

        Examples:
        >>> from swarms.models import OpenAIChat
        >>> from swarms.structs import SequentialWorkflow
        >>> llm = OpenAIChat(openai_api_key="")
        >>> workflow = SequentialWorkflow(max_loops=1)
        >>> workflow.add("What's the weather in miami", llm)
        >>> workflow.add("Create a report on these metrics", llm)
        >>> workflow.update_task("What's the weather in miami", max_tokens=1000)
        >>> workflow.tasks[0].kwargs
        {'max_tokens': 1000}

        """
        try:
            for task in self.tasks:
                if task.description == task:
                    task.kwargs.update(updates)
                    break
            else:
                raise ValueError(
                    f"Task {task} not found in workflow."
                )
        except Exception as error:
            print(
                colored(
                    f"Error updating task in workflow: {error}", "red"
                ),
            )

    def concurrent_run(self):
        """
        Concurrently run the workflow using a pool of workers.

        Examples:
        >>> from swarms.models import OpenAIChat
        >>> from swarms.structs import SequentialWorkflow
        >>> llm = OpenAIChat(openai_api_key="")
        >>> workflow = SequentialWorkflow(max_loops=1)

        """
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures_to_task = {
                    executor.submit(task.run): task
                    for task in self.tasks
                }
                results = []
                for future in concurrent.futures.as_completed(
                    futures_to_task
                ):
                    task = futures_to_task[future]

                    try:
                        result = future.result()
                    except Exception as error:
                        print(f"Error running workflow: {error}")
                    else:
                        results.append(result)
                        print(
                            f"Task {task} completed successfully with"
                            f" result: {result}"
                        )
        except Exception as error:
            print(colored(f"Error running workflow: {error}", "red"))

    def save_workflow_state(
        self,
        filepath: Optional[str] = "sequential_workflow_state.json",
        **kwargs,
    ) -> None:
        """
        Saves the workflow state to a json file.

        Args:
            filepath (str): The path to save the workflow state to.

        Examples:
        >>> from swarms.models import OpenAIChat
        >>> from swarms.structs import SequentialWorkflow
        >>> llm = OpenAIChat(openai_api_key="")
        >>> workflow = SequentialWorkflow(max_loops=1)
        >>> workflow.add("What's the weather in miami", llm)
        >>> workflow.add("Create a report on these metrics", llm)
        >>> workflow.save_workflow_state("sequential_workflow_state.json")
        """
        try:
            filepath = filepath or self.saved_state_filepath

            with open(filepath, "w") as f:
                # Saving the state as a json for simplicuty
                state = {
                    "tasks": [
                        {
                            "description": task.description,
                            "args": task.args,
                            "kwargs": task.kwargs,
                            "result": task.result,
                            "history": task.history,
                        }
                        for task in self.tasks
                    ],
                    "max_loops": self.max_loops,
                }
                json.dump(state, f, indent=4)
        except Exception as error:
            print(
                colored(
                    f"Error saving workflow state: {error}",
                    "red",
                )
            )

    def workflow_bootup(self, **kwargs) -> None:
        """Boots up the workflow."""
        print(
            colored(
                """
                Sequential Workflow Initializing...""",
                "green",
                attrs=["bold", "underline"],
            )
        )

    def workflow_dashboard(self, **kwargs) -> None:
        """
        Displays a dashboard for the workflow.

        Args:
            **kwargs: Additional keyword arguments to pass to the dashboard.

        Examples:
        >>> from swarms.models import OpenAIChat
        >>> from swarms.structs import SequentialWorkflow
        >>> llm = OpenAIChat(openai_api_key="")
        >>> workflow = SequentialWorkflow(max_loops=1)
        >>> workflow.add("What's the weather in miami", llm)
        >>> workflow.add("Create a report on these metrics", llm)
        >>> workflow.workflow_dashboard()

        """
        print(
            colored(
                f"""
                Sequential Workflow Dashboard
                --------------------------------
                Name: {self.name}
                Description: {self.description}
                Tasks: {len(self.tasks)}
                Max Loops: {self.max_loops}
                Autosave: {self.autosave}
                Autosave Filepath: {self.saved_state_filepath}
                Restore Filepath: {self.restore_state_filepath}
                --------------------------------
                Metadata:
                kwargs: {kwargs}
                """,
                "cyan",
                attrs=["bold", "underline"],
            )
        )

    def workflow_shutdown(self, **kwargs) -> None:
        """Shuts down the workflow."""
        print(
            colored(
                """
                Sequential Workflow Shutdown...""",
                "red",
                attrs=["bold", "underline"],
            )
        )

    def add_objective_to_workflow(self, task: str, **kwargs) -> None:
        """Adds an objective to the workflow."""
        try:
                
            print(
                colored(
                    """
                    Adding Objective to Workflow...""",
                    "green",
                    attrs=["bold", "underline"],
                )
            )

            task = Task(
                description=task,
                agent=kwargs["agent"],
                args=list(kwargs["args"]),
                kwargs=kwargs["kwargs"],
            )
            self.tasks.append(task)
        except Exception as error:
            print(
                colored(
                    f"Error adding objective to workflow: {error}",
                    "red",
                )
            )

    def load_workflow_state(
        self, filepath: str = None, **kwargs
    ) -> None:
        """
        Loads the workflow state from a json file and restores the workflow state.

        Args:
            filepath (str): The path to load the workflow state from.

        Examples:
        >>> from swarms.models import OpenAIChat
        >>> from swarms.structs import SequentialWorkflow
        >>> llm = OpenAIChat(openai_api_key="")
        >>> workflow = SequentialWorkflow(max_loops=1)
        >>> workflow.add("What's the weather in miami", llm)
        >>> workflow.add("Create a report on these metrics", llm)
        >>> workflow.save_workflow_state("sequential_workflow_state.json")
        >>> workflow.load_workflow_state("sequential_workflow_state.json")

        """
        try:
                
            filepath = filepath or self.restore_state_filepath

            with open(filepath, "r") as f:
                state = json.load(f)
                self.max_loops = state["max_loops"]
                self.tasks = []
                for task_state in state["tasks"]:
                    task = Task(
                        description=task_state["description"],
                        agent=task_state["agent"],
                        args=task_state["args"],
                        kwargs=task_state["kwargs"],
                        result=task_state["result"],
                        history=task_state["history"],
                    )
                    self.tasks.append(task)
        except Exception as error:
            print(
                colored(
                    f"Error loading workflow state: {error}",
                    "red",
                )
            )

    def run(self) -> None:
        """
        Run the workflow.

        Raises:
            ValueError: If a Agent instance is used as a task and the 'task' argument is not provided.

        """
        try:
            self.workflow_bootup()
            for _ in range(self.max_loops):
                for task in self.tasks:
                    # Check if the current task can be executed
                    if task.result is None:
                        # Check if the agent is a Agent and a 'task' argument is needed
                        if isinstance(task.agent, Agent):
                            # Ensure that 'task' is provided in the kwargs
                            if "task" not in task.kwargs:
                                raise ValueError(
                                    "The 'task' argument is required"
                                    " for the Agent agent execution"
                                    f" in '{task.description}'"
                                )
                            # Separate the 'task' argument from other kwargs
                            flow_task_arg = task.kwargs.pop("task")
                            task.result = task.agent.run(
                                flow_task_arg,
                                *task.args,
                                **task.kwargs,
                            )
                        else:
                            # If it's not a Agent instance, call the agent directly
                            task.result = task.agent(
                                *task.args, **task.kwargs
                            )

                        # Pass the result as an argument to the next task if it exists
                        next_task_index = self.tasks.index(task) + 1
                        if next_task_index < len(self.tasks):
                            next_task = self.tasks[next_task_index]
                            if isinstance(next_task.agent, Agent):
                                # For Agent flows, 'task' should be a keyword argument
                                next_task.kwargs["task"] = task.result
                            else:
                                # For other callable flows, the result is added to args
                                next_task.args.insert(0, task.result)

                        # Autosave the workflow state
                        if self.autosave:
                            self.save_workflow_state(
                                "sequential_workflow_state.json"
                            )
        except Exception as e:
            print(
                colored(
                    (
                        "Error initializing the Sequential workflow:"
                        f" {e} try optimizing your inputs like the"
                        " agent class and task description"
                    ),
                    "red",
                    attrs=["bold", "underline"],
                )
            )

    async def arun(self) -> None:
        """
        Asynchronously run the workflow.

        Raises:
            ValueError: If a Agent instance is used as a task and the 'task' argument is not provided.

        """
        try:
            for _ in range(self.max_loops):
                for task in self.tasks:
                    # Check if the current task can be executed
                    if task.result is None:
                        # Check if the agent is a Agent and a 'task' argument is needed
                        if isinstance(task.agent, Agent):
                            # Ensure that 'task' is provided in the kwargs
                            if "task" not in task.kwargs:
                                raise ValueError(
                                    "The 'task' argument is required for"
                                    " the Agent agent execution in"
                                    f" '{task.description}'"
                                )
                            # Separate the 'task' argument from other kwargs
                            flow_task_arg = task.kwargs.pop("task")
                            task.result = await task.agent.arun(
                                flow_task_arg, *task.args, **task.kwargs
                            )
                        else:
                            # If it's not a Agent instance, call the agent directly
                            task.result = await task.agent(
                                *task.args, **task.kwargs
                            )

                        # Pass the result as an argument to the next task if it exists
                        next_task_index = self.tasks.index(task) + 1
                        if next_task_index < len(self.tasks):
                            next_task = self.tasks[next_task_index]
                            if isinstance(next_task.agent, Agent):
                                # For Agent flows, 'task' should be a keyword argument
                                next_task.kwargs["task"] = task.result
                            else:
                                # For other callable flows, the result is added to args
                                next_task.args.insert(0, task.result)

                        # Autosave the workflow state
                        if self.autosave:
                            self.save_workflow_state(
                                "sequential_workflow_state.json"
                            )
        except Exception as e:
            print(
                colored(
                    (
                        "Error initializing the Sequential workflow:"
                        f" {e} try optimizing your inputs like the"
                        " agent class and task description"
                    ),
                    "red",
                    attrs=["bold", "underline"],
                )
            )