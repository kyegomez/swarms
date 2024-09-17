import json
import sched
import time
from datetime import datetime
from typing import Any, Callable, ClassVar, Dict, List, Union

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.omni_agent_types import AgentType
from swarms.utils.loguru_logger import logger
from typing import Optional


class Task(BaseModel):
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
    >>> from swarm_models import OpenAIChat
    >>> agent = Agent(llm=OpenAIChat(openai_api_key=""), max_loops=1, dashboard=False)
    >>> task = Task(description="What's the weather in miami", agent=agent)
    >>> task.run()

    >>> task.result

    """

    name: Optional[str] = "Task"
    description: Optional[str] = (
        "A task is a unit of work that needs to be completed for a workflow to progress."
    )
    agent: Optional[Union[Callable, Agent, AgentType]] = Field(
        None,
        description="Agent or callable object to run the task",
    )
    result: Optional[Any] = None
    history: List[Any] = Field(default_factory=list)
    schedule_time: Optional[datetime] = Field(
        None,
        description="Time to schedule the task",
    )
    scheduler: ClassVar[sched.scheduler] = sched.scheduler(
        time.time, time.sleep
    )
    trigger: Optional[Callable] = Field(
        None,
        description="Trigger to run the task",
    )
    action: Optional[Callable] = Field(
        None,
        description="Action to run the task",
    )
    condition: Optional[Callable] = Field(
        None,
        description="Condition to run the task",
    )
    priority: Optional[int] = Field(
        0.4,
        description="Priority of the task",
    )
    dependencies: List["Task"] = Field(default_factory=list)
    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

        # We need to check that the agent exists

    def step(self, task: str = None, *args, **kwargs):
        """
        Execute the task by calling the agent or model with the arguments and
        keyword arguments. You can add images to the agent by passing the
        path to the image as a keyword argument.


        Examples:
        >>> from swarms.structs import Task, Agent
        >>> from swarm_models import OpenAIChat
        >>> agent = Agent(llm=OpenAIChat(openai_api_key=""), max_loops=1, dashboard=False)
        >>> task = Task(description="What's the weather in miami", agent=agent)
        >>> task.run()
        >>> task.result

        """

        logger.info(f"Running task: {task}")

        # Check dependencies
        if not self.check_dependency_completion():
            logger.info(
                f"Task {self.description} is waiting for dependencies to complete"
            )
            return None

        # Check the condition before executing the task
        if self.condition is not None:
            try:
                condition_result = self.condition()
                if not condition_result:
                    logger.info(
                        f"Completion not met for the task: {task} Skipping execution"
                    )
                    return None
            except Exception as error:
                logger.error(f"[ERROR][Task] {error}")
                return None

        # Execute the task
        if self.trigger is None or self.trigger():
            try:
                logger.info(f"Executing task: {task}")
                self.result = self.agent.run(task, *args, **kwargs)

                # Ensure the result is either a string or a dict
                if isinstance(self.result, str):
                    logger.info(f"Task result: {self.result}")
                elif isinstance(self.result, dict):
                    logger.info(f"Task result: {self.result}")
                else:
                    logger.error(
                        "Task result must be either a string or a dict"
                    )

                # Add the result to the history
                self.history.append(self.result)

                # If an action is specified, execute it
                if self.action is not None:
                    try:
                        logger.info(
                            f"Executing action for task: {task}"
                        )
                        self.action()
                    except Exception as error:
                        logger.error(f"[ERROR][Task] {error}")
            except Exception as error:
                logger.error(f"[ERROR][Task] {error}")
        else:
            logger.info(f"Task {task} is not triggered")

    def run(self, task: str = None, *args, **kwargs):
        now = datetime.now()

        # If the task is scheduled for the future, schedule it
        if self.schedule_time and self.schedule_time > now:
            delay = (self.schedule_time - now).total_seconds()
            logger.info(
                f"Scheduling task: {self.description} for {self.schedule_time}"
            )
            self.scheduler.enter(
                delay,
                1,
                self.step,
                argument=(task, args, kwargs),
            )
            self.scheduler.run()

            # We need to return the result
        else:
            # If no scheduling or the time has already passed run the task
            return self.step(task, *args, **kwargs)

    def handle_scheduled_task(self):
        """
        Handles the execution of a scheduled task.

        If the schedule time is not set or has already passed, the task is executed immediately.
        Otherwise, the task is scheduled to be executed at the specified schedule time.
        """
        logger.info(
            f"[INFO][Task] Handling scheduled task: {self.description}"
        )
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

                result = (
                    task.result if task.result is not None else ""
                )

                # Add the context of the task to the conversation
                new_context.add(
                    task.agent.agent_name, f"{description} {result}"
                )

        elif task:
            description = (
                task.description
                if task.description is not None
                else ""
            )
            result = task.result if task.result is not None else ""
            new_context.add(
                task.agent.agent_name, f"{description} {result}"
            )

        prompt = new_context.return_history_as_string()

        # Add to history
        return self.history.append(prompt)

    def to_dict(self):
        """
        Convert the task to a dictionary.

        Returns:
            dict: The task as a dictionary.
        """
        return self.model_dump_json(indent=4)

    def save_to_file(self, file_path: str):
        """
        Save the task to a file.

        Args:
            file_path (str): The path to the file to save the task to.
        """
        with open(file_path, "w") as file:
            file.write(self.to_json(indent=4))

    @classmethod
    def load_from_file(cls, file_path: str):
        """
        Load a task from a file.

        Args:
            file_path (str): The path to the file to load the task from.

        Returns:
            Task: The task loaded from the file.
        """
        with open(file_path, "r") as file:
            task_dict = json.load(file)
            return Task(**task_dict)

    def schedule_task_with_sched(
        function: Callable, run_date: datetime
    ) -> None:
        now = datetime.now()

        if run_date <= now:
            raise ValueError("run_date must be in the future")

        # Calculate the delay in seconds
        delay = (run_date - now).total_seconds()

        scheduler = sched.scheduler(time.time, time.sleep)

        # Schedule the function
        scheduler.enter(delay, 1, function)

        # Start the scheduler
        scheduler.run(delay, 1, function)

        # Start the scheduler
        logger.info(f"Task scheduled for {run_date}")
        scheduler.run()

        return None
