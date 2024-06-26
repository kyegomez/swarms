import sched
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Union

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.loguru_logger import logger
from swarms.structs.omni_agent_types import AgentType


@dataclass
class Task:
    """
    Task class for running a task in a sequential workflow.

    Attributes:
        description (str): Description of the task.
        agent (Union[Callable, Agent]): Agent or callable object to run the task.
        args (List[Any]): Arguments to pass to the agent or callable object.
        kwargs (Dict[str, Any]): Keyword arguments to pass to the agent or callable object.
        result (Any): Result of the task.
        history (List[Any]): History of the task.
        schedule_time (datetime): Time to schedule the task.
        scheduler (sched.scheduler): Scheduler to schedule the task.
        trigger (Callable): Trigger to run the task.
        action (Callable): Action to run the task.
        condition (Callable): Condition to run the task.
        priority (int): Priority of the task.
        dependencies (List[Task]): List of tasks that need to be completed before this task can be executed.

    Methods:
        execute: Execute the task by calling the agent or model with the arguments and keyword arguments.
        handle_scheduled_task: Handles the execution of a scheduled task.
        set_trigger: Sets the trigger for the task.
        set_action: Sets the action for the task.
        set_condition: Sets the condition for the task.
        is_completed: Checks whether the task has been completed.
        add_dependency: Adds a task to the list of dependencies.
        set_priority: Sets the priority of the task.
        check_dependency_completion: Checks whether all the dependencies have been completed.


    Examples:
    >>> from swarms.structs import Task, Agent
    >>> from swarms.models import OpenAIChat
    >>> agent = Agent(llm=OpenAIChat(openai_api_key=""), max_loops=1, dashboard=False)
    >>> task = Task(description="What's the weather in miami", agent=agent)
    >>> task.execute()
    >>> task.result

    """

    agent: Union[Callable, Agent, AgentType] = None
    description: str = None
    result: Any = None
    history: List[Any] = field(default_factory=list)
    schedule_time: datetime = None
    scheduler = sched.scheduler(time.time, time.sleep)
    trigger: Callable = None
    action: Callable = None
    condition: Callable = None
    priority: int = 0
    dependencies: List["Task"] = field(default_factory=list)
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def execute(self, *args, **kwargs):
        """
        Execute the task by calling the agent or model with the arguments and
        keyword arguments. You can add images to the agent by passing the
        path to the image as a keyword argument.


        Examples:
        >>> from swarms.structs import Task, Agent
        >>> from swarms.models import OpenAIChat
        >>> agent = Agent(llm=OpenAIChat(openai_api_key=""), max_loops=1, dashboard=False)
        >>> task = Task(description="What's the weather in miami", agent=agent)
        >>> task.execute()
        >>> task.result

        """
        logger.info(f"[INFO][Task] Executing task: {self.description}")
        task = self.description
        try:
            if isinstance(self.agent, Agent):
                if self.condition is None or self.condition():
                    self.result = self.agent.run(
                        task=task,
                        *args,
                        **kwargs,
                    )
                    self.history.append(self.result)

                    if self.action is not None:
                        self.action()
            else:
                self.result = self.agent.run(*self.args, **self.kwargs)

            self.history.append(self.result)
        except Exception as error:
            logger.error(f"[ERROR][Task] {error}")

    def run(self, *args, **kwargs):
        self.execute(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.execute(*args, **kwargs)

    def handle_scheduled_task(self):
        """
        Handles the execution of a scheduled task.

        If the schedule time is not set or has already passed, the task is executed immediately.
        Otherwise, the task is scheduled to be executed at the specified schedule time.
        """
        logger.info("[INFO][Task] Handling scheduled task")
        try:
            if (
                self.schedule_time is None
                or self.schedule_time <= datetime.now()
            ):
                self.execute()

            else:
                delay = (
                    self.schedule_time - datetime.now()
                ).total_seconds()
                self.scheduler.enter(delay, 1, self.execute)
                self.scheduler_run()
        except Exception as error:
            logger.error(f"[ERROR][Task] {error}")

    def set_trigger(self, trigger: Callable):
        """
        Sets the trigger for the task.

        Args:
            trigger (Callable): The trigger to set.
        """
        self.trigger = trigger

    def set_action(self, action: Callable):
        """
        Sets the action for the task.

        Args:
            action (Callable): The action to set.
        """
        self.action = action

    def set_condition(self, condition: Callable):
        """
        Sets the condition for the task.

        Args:
            condition (Callable): The condition to set.
        """
        self.condition = condition

    def is_completed(self):
        """Is the task completed?

        Returns:
            _type_: _description_
        """
        return self.result is not None

    def add_dependency(self, task):
        """Adds a task to the list of dependencies.

        Args:
            task (_type_): _description_
        """
        self.dependencies.append(task)

    def set_priority(self, priority: int):
        """Sets the priority of the task.

        Args:
            priority (int): _description_
        """
        self.priority = priority

    def check_dependency_completion(self):
        """
        Checks whether all the dependencies have been completed.

        Returns:
            bool: True if all the dependencies have been completed, False otherwise.
        """
        logger.info("[INFO][Task] Checking dependency completion")
        try:
            for task in self.dependencies:
                if not task.is_completed():
                    return False
        except Exception as error:
            logger.error(
                f"[ERROR][Task][check_dependency_completion] {error}"
            )

    def context(
        self,
        task: "Task" = None,
        context: List["Task"] = None,
        *args,
        **kwargs,
    ):
        """
        Set the context for the task.

        Args:
            context (str): The context to set.
        """
        # For sequential workflow, sequentially add the context of the previous task in the list
        new_context = Conversation(time_enabled=True, *args, **kwargs)

        if context:
            for task in context:
                description = (
                    task.description
                    if task.description is not None
                    else ""
                )

                result = task.result if task.result is not None else ""

                # Add the context of the task to the conversation
                new_context.add(
                    task.agent.agent_name, f"{description} {result}"
                )

        elif task:
            description = (
                task.description if task.description is not None else ""
            )
            result = task.result if task.result is not None else ""
            new_context.add(
                task.agent.agent_name, f"{description} {result}"
            )

        prompt = new_context.return_history_as_string()

        # Add to history
        return self.history.append(prompt)
