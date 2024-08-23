import uuid
import csv
import datetime
import os
import queue
import threading
from typing import List, Union

from pydantic import BaseModel

from swarms.structs.agent import Agent
from swarms.utils.file_processing import create_file_in_folder
from swarms.utils.loguru_logger import logger

time = datetime.datetime.now().isoformat()


class AgentOutput(BaseModel):
    agent_name: str
    task: str
    result: str
    timestamp: str


class SwarmRunMetadata(BaseModel):
    run_id: str
    name: str
    description: str
    agents: List[str]
    start_time: str
    end_time: str
    tasks_completed: int
    outputs: List[AgentOutput]


class SpreadSheetSwarm:
    """
    A swarm that processes tasks from a queue using multiple agents on different threads.

    Attributes:
        name (str): The name of the swarm.
        description (str): The description of the swarm.
        agents (Union[Agent, List[Agent]]): The agents participating in the swarm.
        autosave_on (bool): Flag indicating whether autosave is enabled.
        save_file_path (str): The file path to save the swarm data.
        task_queue (queue.Queue): The queue to store tasks.
        lock (threading.Lock): The lock used for thread synchronization.
        metadata (SwarmRunMetadata): The metadata for the swarm run.
    """

    def __init__(
        self,
        name: str = "Spreadsheet-Swarm",
        description: str = "A swarm that processes tasks from a queue using multiple agents on different threads.",
        agents: Union[Agent, List[Agent]] = [],
        autosave_on: bool = True,
        save_file_path: str = "spreedsheet_swarm.csv",
        run_all_agents: bool = True,
        repeat_count: int = 1,
        workspace_dir: str = os.getenv("WORKSPACE_DIR"),
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.autosave_on = autosave_on
        self.save_file_path = save_file_path
        self.run_all_agents = run_all_agents
        self.repeat_count = repeat_count
        self.workspace_dir = workspace_dir
        self.task_queue = queue.Queue()
        self.lock = threading.Lock()

        # Metadata
        if isinstance(agents, Agent):
            agents = [agents]

        # Metadata
        self.metadata = SwarmRunMetadata(
            run_id=f"spreadsheet_swarm_run_{time}",
            name=name,
            description=description,
            agents=[agent.name for agent in agents],
            start_time=time,
            end_time="",
            tasks_completed=0,
            outputs=[],
        )

        # Check the reliability of the swarm
        self.reliability_check()

    def reliability_check(self):
        logger.info("Checking the reliability of the swarm...")

        if not self.agents:
            raise ValueError("No agents are provided.")
        if not self.save_file_path:
            raise ValueError("No save file path is provided.")

        logger.info("Swarm reliability check passed.")
        logger.info("Swarm is ready to run.")

    def run(self, task: str, *args, **kwargs):
        """
        Run the swarm with the given task.

        Args:
            task (str): The task to run.
            run_all_agents (bool): Whether to run all agents.
            repeat_count (int): The number of times to repeat the task.
        """
        self.metadata.start_time = time

        # # If the task is a list, add each task to the queue
        # if isinstance(task, list):
        #     task = [task]

        # If run_all_agents is True, run all agents
        # if self.run_all_agents:
        #     for agent in self.agents:
        #         for t in task:
        #             result = agent.run(t, *args, **kwargs)
        #             self._track_output(agent, t, result)
        # else:
        #     agent = self.agents[0]
        #     for t in task:
        #         for _ in range(self.repeat_count):
        #             result = agent.run(t, *args, **kwargs)
        #             self._track_output(agent, t, result)

        # for i in range(self.repeat_count):
        #     results = run_agents_concurrently(self.agents, task)

        #     for agent, result in zip(self.agents, results):
        #         self._track_output(agent, task, result)

        for i in range(self.repeat_count):

            for agent in self.agents:
                result = agent.run(task, *args, **kwargs)
                self._track_output(agent, task, result)

        # Set the end time
        self.metadata.end_time = time

        # Save the metadata to a CSV file
        self._save_to_csv()

        # Export the metadata to JSON
        if self.autosave_on:
            self.data_to_json_file()

    def export_to_json(self):
        """
        export the metadata to a JSON file.

        Returns:
            str: The JSON representation of the metadata.


        """
        return self.metadata.model_dump_json(indent=4)

    def data_to_json_file(self):
        out = str(self.export_to_json())

        # Save the JSON to a file
        create_file_in_folder(
            folder_path=f"{self.workspace_dir}/Spreedsheet-Swarm/{self.name}",
            file_name=f"spreedsheet-swarm-{self.metadata.run_id}_metadata.json",
            content=out,
        )

    def _track_output(self, agent: Agent, task: str, result: str):
        """
        Track the output of the agent.

        Args:
            agent (Agent): The agent.
            task (str): The task.
            result (str): The result.
        """
        self.metadata.tasks_completed += 1
        self.metadata.outputs.append(
            AgentOutput(
                agent_name=agent.agent_name,
                task=task,
                result=result,
                timestamp=time,
            )
        )

    def _save_to_csv(self):
        """
        Save the swarm metadata to a CSV file.
        """
        logger.info(f"Saving swarm metadata to: {self.save_file_path}")
        run_id = uuid.uuid4()  # Generate a unique run ID

        # Check if the file exists
        file_exists = os.path.isfile(self.save_file_path)

        with open(self.save_file_path, "a", newline="") as file:
            writer = csv.writer(file)

            # If the file didn't exist, write the header
            if not file_exists:
                writer.writerow(
                    ["Run ID", "Agent Name", "Task", "Result", "Timestamp"]
                )

            for output in self.metadata.outputs:
                # Log the task and result before writing them to the CSV file
                logger.info(
                    f"Task: {output.task}, Result: {output.result}"
                )
                writer.writerow(
                    [
                        str(run_id),
                        output.agent_name,
                        output.task,
                        output.result,
                        output.timestamp,
                    ]
                )
