import json
from typing import List, Optional

from pydantic import BaseModel, Field, Json, model_validator

from swarms.structs.agent import Agent
from swarms.structs.task import Task


class Team(BaseModel):
    """
    Class that represents a group of agents, how they should work together and
    their tasks.

    Attributes:
        tasks (Optional[List[Task]]): List of tasks.
        agents (Optional[List[Agent]]): List of agents in this Team.
        architecture (str): Architecture that the Team will follow. Default is "sequential".
        verbose (bool): Verbose mode for the Agent Execution. Default is False.
        config (Optional[Json]): Configuration of the Team. Default is None.
    """

    tasks: Optional[List[Task]] = Field(
        None, description="List of tasks"
    )
    agents: Optional[List[Agent]] = Field(
        None, description="List of agents in this Team."
    )
    architecture = Field(
        description="architecture that the Team will follow.",
        default="sequential",
    )
    verbose: bool = Field(
        description="Verbose mode for the Agent Execution",
        default=False,
    )
    config: Optional[Json] = Field(
        description="Configuration of the Team.", default=None
    )

    @model_validator(mode="before")
    @classmethod
    def check_config(_cls, values):
        if not values.get("config") and (
            not values.get("agents") and not values.get("tasks")
        ):
            raise ValueError(
                "Either agents and task need to be set or config."
            )

        if values.get("config"):
            config = json.loads(values.get("config"))
            if not config.get("agents") or not config.get("tasks"):
                raise ValueError(
                    "Config should have agents and tasks."
                )

            values["agents"] = [
                Agent(**agent) for agent in config["agents"]
            ]

            tasks = []
            for task in config["tasks"]:
                task_agent = [
                    agt
                    for agt in values["agents"]
                    if agt.role == task["agent"]
                ][0]
                del task["agent"]
                tasks.append(Task(**task, agent=task_agent))

            values["tasks"] = tasks
        return values

    def run(self) -> str:
        """
        Kickoff the Team to work on its tasks.

        Returns:
            output (List[str]): Output of the Team for each task.
        """
        if self.architecture == "sequential":
            return self.__sequential_loop()

    def __sequential_loop(self) -> str:
        """
        Loop that executes the sequential architecture.

        Returns:
            output (str): Output of the Team.
        """
        task_outcome = None
        for task in self.tasks:
            # Add delegation tools to the task if the agent allows it
            # if task.agent.allow_delegation:
            #     tools = AgentTools(agents=self.agents).tools()
            #     task.tools += tools

            self.__log(f"\nWorking Agent: {task.agent.role}")
            self.__log(f"Starting Task: {task.description} ...")

            task_outcome = task.execute(task_outcome)

            self.__log(f"Task output: {task_outcome}")

        return task_outcome

    def __log(self, message):
        if self.verbose:
            print(message)
