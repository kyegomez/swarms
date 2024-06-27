import json
from typing import Any, Dict, List, Optional

from termcolor import colored

from swarms.structs.agent import Agent
from swarms.structs.base_structure import BaseStructure
from swarms.structs.task import Task
from swarms.utils.loguru_logger import logger


class BaseWorkflow(BaseStructure):
    """
    Base class for defining a workflow.

    Args:
        agents (List[Agent], optional): A list of agents participating in the workflow. Defaults to None.
        task_pool (List[Task], optional): A list of tasks in the workflow. Defaults to None.
        models (List[Any], optional): A list of models used in the workflow. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        agents (List[Agent]): A list of agents participating in the workflow.
        task_pool (List[Task]): A list of tasks in the workflow.
        models (List[Any]): A list of models used in the workflow.

    Methods:
        add_task: Adds a task or a list of tasks to the task pool.
        add_agent: Adds an agent to the workflow.
        run: Abstract method to run the workflow.
        reset: Resets the workflow by clearing the results of each task.
        get_task_results: Returns the results of each task in the workflow.
        remove_task: Removes a task from the workflow.
        update_task: Updates the arguments of a task in the workflow.
        delete_task: Deletes a task from the workflow.
        save_workflow_state: Saves the workflow state to a json file.
        add_objective_to_workflow: Adds an objective to the workflow.
        load_workflow_state: Loads the workflow state from a json file and restores the workflow state.
    """

    def __init__(
        self,
        agents: List[Agent] = None,
        task_pool: List[Task] = None,
        models: List[Any] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agents = agents
        self.task_pool = task_pool
        self.models = models
        self.task_pool = []
        self.agent_pool = []

        # Logging
        logger.info("Number of agents activated:")
        if self.agents:
            logger.info(f"Agents: {len(self.agents)}")
        else:
            logger.info("No agents activated.")

        if self.task_pool:
            logger.info(f"Task Pool Size: {len(self.task_pool)}")
        else:
            logger.info("Task Pool is empty.")

    def add_task(
        self,
        task: Task = None,
        tasks: List[Task] = None,
        *args,
        **kwargs,
    ):
        """
        Adds a task or a list of tasks to the task pool.

        Args:
            task (Task, optional): A single task to add. Defaults to None.
            tasks (List[Task], optional): A list of tasks to add. Defaults to None.

        Raises:
            ValueError: If neither task nor tasks are provided.
        """
        if task:
            self.task_pool.append(task)
        elif tasks:
            self.task_pool.extend(tasks)
        else:
            raise ValueError("You must provide a task or a list of tasks")

    def add_agent(self, agent: Agent, *args, **kwargs):
        return self.agent_pool(agent)

    def run(self):
        """
        Abstract method to run the workflow.
        """
        ...

    def __sequential_loop(self):
        """
        Abstract method for the sequential loop.
        """
        # raise NotImplementedError("You must implement this method")
        ...

    def __log(self, message: str):
        """
        Logs a message if verbose mode is enabled.

        Args:
            message (str): The message to log.
        """
        if self.verbose:
            print(message)

    def __str__(self):
        return f"Workflow with {len(self.task_pool)} tasks"

    def __repr__(self):
        return f"Workflow with {len(self.task_pool)} tasks"

    def reset(self) -> None:
        """Resets the workflow by clearing the results of each task."""
        try:
            for task in self.tasks:
                task.result = None
        except Exception as error:
            print(
                colored(f"Error resetting workflow: {error}", "red"),
            )

    def get_task_results(self) -> Dict[str, Any]:
        """
        Returns the results of each task in the workflow.

        Returns:
            Dict[str, Any]: The results of each task in the workflow
        """
        try:
            return {task.description: task.result for task in self.tasks}
        except Exception as error:
            print(
                colored(f"Error getting task results: {error}", "red"),
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
                    f"Error removing task from workflow: {error}",
                    "red",
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
                raise ValueError(f"Task {task} not found in workflow.")
        except Exception as error:
            print(
                colored(
                    f"Error updating task in workflow: {error}", "red"
                ),
            )

    def delete_task(self, task: str) -> None:
        """
        Delete a task from the workflow.

        Args:
            task (str): The description of the task to delete.

        Raises:
            ValueError: If the task is not found in the workflow.

        Examples:
        >>> from swarms.models import OpenAIChat
        >>> from swarms.structs import SequentialWorkflow
        >>> llm = OpenAIChat(openai_api_key="")
        >>> workflow = SequentialWorkflow(max_loops=1)
        >>> workflow.add("What's the weather in miami", llm)
        >>> workflow.add("Create a report on these metrics", llm)
        >>> workflow.delete_task("What's the weather in miami")
        >>> workflow.tasks
        [Task(description='Create a report on these metrics', agent=Agent(llm=OpenAIChat(openai_api_key=''), max_loops=1, dashboard=False), args=[], kwargs={}, result=None, history=[])]
        """
        try:
            for task in self.tasks:
                if task.description == task:
                    self.tasks.remove(task)
                    break
            else:
                raise ValueError(f"Task {task} not found in workflow.")
        except Exception as error:
            print(
                colored(
                    f"Error deleting task from workflow: {error}",
                    "red",
                ),
            )

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

    def load_workflow_state(self, filepath: str = None, **kwargs) -> None:
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

            with open(filepath) as f:
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
                task_pool: {len(self.task_pool)}
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

    def workflow_bootup(self, **kwargs) -> None:
        """
        Workflow bootup.

        """
        print(
            colored(
                """
                Sequential Workflow Initializing...""",
                "green",
                attrs=["bold", "underline"],
            )
        )
