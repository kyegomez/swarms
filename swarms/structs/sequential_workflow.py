import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from termcolor import colored

from swarms.structs.task import Task
from swarms.utils.logger import logger


# SequentialWorkflow class definition using dataclasses
@dataclass
class SequentialWorkflow:
    """
    SequentialWorkflow class for running a sequence of task_pool using N number of autonomous agents.

    Args:
        max_loops (int): The maximum number of times to run the workflow.
        dashboard (bool): Whether to display the dashboard for the workflow.


    Attributes:
        task_pool (List[Task]): The list of task_pool to execute.
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
    >>> workflow.task_pool

    """

    name: str = None
    description: str = None
    task_pool: List[Task] = field(default_factory=list)
    max_loops: int = 1
    autosave: bool = False
    saved_state_filepath: Optional[str] = (
        "sequential_workflow_state.json"
    )
    restore_state_filepath: Optional[str] = None
    dashboard: bool = False

    def add(
        self,
        task: Optional[Task] = None,
        tasks: Optional[List[Task]] = None,
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
            # If the agent is a Task instance, we include the task in kwargs for Agent.run()
            # Append the task to the task_pool list
            if task:
                self.task_pool.append(task)
                logger.info(
                    f"[INFO][SequentialWorkflow] Added task {task} to"
                    " workflow"
                )
            elif tasks:
                for task in tasks:
                    self.task_pool.append(task)
                    logger.info(
                        "[INFO][SequentialWorkflow] Added task"
                        f" {task} to workflow"
                    )
            else:
                if task and tasks is not None:
                    # Add the task and list of tasks to the task_pool at the same time
                    self.task_pool.append(task)
                    for task in tasks:
                        self.task_pool.append(task)

        except Exception as error:
            logger.error(
                colored(
                    f"Error adding task to workflow: {error}", "red"
                ),
            )

    def reset_workflow(self) -> None:
        """Resets the workflow by clearing the results of each task."""
        try:
            for task in self.task_pool:
                task.result = None
                logger.info(
                    f"[INFO][SequentialWorkflow] Reset task {task} in"
                    " workflow"
                )
        except Exception as error:
            logger.error(
                colored(f"Error resetting workflow: {error}", "red"),
            )

    def get_task_results(self) -> Dict[str, Any]:
        """
        Returns the results of each task in the workflow.

        Returns:
            Dict[str, Any]: The results of each task in the workflow
        """
        try:
            return {
                task.description: task.result
                for task in self.task_pool
            }
        except Exception as error:
            logger.error(
                colored(
                    f"Error getting task results: {error}", "red"
                ),
            )

    def remove_task(self, task: Task) -> None:
        """Remove task_pool from sequential workflow"""
        try:
            self.task_pool.remove(task)
            logger.info(
                f"[INFO][SequentialWorkflow] Removed task {task} from"
                " workflow"
            )
        except Exception as error:
            logger.error(
                colored(
                    f"Error removing task from workflow: {error}",
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
                    "task_pool": [
                        {
                            "description": task.description,
                            "args": task.args,
                            "kwargs": task.kwargs,
                            "result": task.result,
                            "history": task.history,
                        }
                        for task in self.task_pool
                    ],
                    "max_loops": self.max_loops,
                }
                json.dump(state, f, indent=4)

            logger.info(
                "[INFO][SequentialWorkflow] Saved workflow state to"
                f" {filepath}"
            )
        except Exception as error:
            logger.error(
                colored(
                    f"Error saving workflow state: {error}",
                    "red",
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

            with open(filepath) as f:
                state = json.load(f)
                self.max_loops = state["max_loops"]
                self.task_pool = []
                for task_state in state["task_pool"]:
                    task = Task(
                        description=task_state["description"],
                        agent=task_state["agent"],
                        args=task_state["args"],
                        kwargs=task_state["kwargs"],
                        result=task_state["result"],
                        history=task_state["history"],
                    )
                    self.task_pool.append(task)

            print(
                "[INFO][SequentialWorkflow] Loaded workflow state"
                f" from {filepath}"
            )
        except Exception as error:
            logger.error(
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
            loops = 0
            while loops < self.max_loops:
                for i in range(len(self.task_pool)):
                    task = self.task_pool[i]
                    # Check if the current task can be executed
                    if task.result is None:
                        # Get the inputs for the current task
                        task.context(task)

                        result = task.execute()

                        # Pass the inputs to the next task
                        if i < len(self.task_pool) - 1:
                            next_task = self.task_pool[i + 1]
                            next_task.description = result

                        # Execute the current task
                        task.execute()

                        # Autosave the workflow state
                        if self.autosave:
                            self.save_workflow_state(
                                "sequential_workflow_state.json"
                            )

                self.workflow_shutdown()
                loops += 1
        except Exception as e:
            logger.error(
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
