import asyncio
import csv
import datetime
import os
import uuid
from typing import Dict, List, Union

import aiofiles
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.telemetry.main import log_agent_data
from swarms.utils.file_processing import create_file_in_folder
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="spreadsheet_swarm")

time = datetime.datetime.now().isoformat()
uuid_hex = uuid.uuid4().hex

# --------------- NEW CHANGE START ---------------
# Format time variable to be compatible across operating systems
formatted_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# --------------- NEW CHANGE END ---------------


class AgentConfig(BaseModel):
    """Configuration for an agent loaded from CSV"""

    agent_name: str
    description: str
    system_prompt: str
    task: str


class AgentOutput(BaseModel):
    agent_name: str
    task: str
    result: str
    timestamp: str


class SwarmRunMetadata(BaseModel):
    run_id: str = Field(
        default_factory=lambda: f"spreadsheet_swarm_run_{uuid_hex}"
    )
    name: str
    description: str
    agents: List[str]
    start_time: str = Field(
        default_factory=lambda: time,
        description="The start time of the swarm run.",
    )
    end_time: str
    tasks_completed: int
    outputs: List[AgentOutput]
    number_of_agents: int = Field(
        ...,
        description="The number of agents participating in the swarm.",
    )


class SpreadSheetSwarm(BaseSwarm):
    """
    A swarm that processes tasks concurrently using multiple agents.

    Args:
        name (str, optional): The name of the swarm. Defaults to "Spreadsheet-Swarm".
        description (str, optional): The description of the swarm. Defaults to "A swarm that processes tasks concurrently using multiple agents.".
        agents (Union[Agent, List[Agent]], optional): The agents participating in the swarm. Defaults to an empty list.
        autosave_on (bool, optional): Whether to enable autosave of swarm metadata. Defaults to True.
        save_file_path (str, optional): The file path to save the swarm metadata as a CSV file. Defaults to "spreedsheet_swarm.csv".
        max_loops (int, optional): The number of times to repeat the swarm tasks. Defaults to 1.
        workspace_dir (str, optional): The directory path of the workspace. Defaults to the value of the "WORKSPACE_DIR" environment variable.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        name: str = "Spreadsheet-Swarm",
        description: str = "A swarm that that processes tasks concurrently using multiple agents and saves the metadata to a CSV file.",
        agents: Union[Agent, List[Agent]] = [],
        autosave_on: bool = True,
        save_file_path: str = None,
        max_loops: int = 1,
        workspace_dir: str = os.getenv("WORKSPACE_DIR"),
        load_path: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            agents=agents if isinstance(agents, list) else [agents],
            *args,
            **kwargs,
        )
        self.name = name
        self.description = description
        self.save_file_path = save_file_path
        self.autosave_on = autosave_on
        self.max_loops = max_loops
        self.workspace_dir = workspace_dir
        self.load_path = load_path
        self.agent_configs: Dict[str, AgentConfig] = {}

        # --------------- NEW CHANGE START ---------------
        # The save_file_path now uses the formatted_time and uuid_hex
        self.save_file_path = (
            f"spreadsheet_swarm_run_id_{uuid_hex}.csv"
        )
        # --------------- NEW CHANGE END ---------------

        self.metadata = SwarmRunMetadata(
            run_id=f"spreadsheet_swarm_run_{time}",
            name=name,
            description=description,
            agents=[agent.name for agent in agents],
            start_time=time,
            end_time="",
            tasks_completed=0,
            outputs=[],
            number_of_agents=len(agents),
        )

        self.reliability_check()

    def reliability_check(self):
        """
        Check the reliability of the swarm.

        Raises:
            ValueError: If no agents are provided or no save file path is provided.
        """
        logger.info("Checking the reliability of the swarm...")

        # if not self.agents:
        #     raise ValueError("No agents are provided.")
        # if not self.save_file_path:
        #     raise ValueError("No save file path is provided.")
        if not self.max_loops:
            raise ValueError("No max loops are provided.")

        logger.info("Swarm reliability check passed.")
        logger.info("Swarm is ready to run.")

    async def _load_from_csv(self):
        """
        Load agent configurations from a CSV file.
        Expected CSV format: agent_name,description,system_prompt,task

        Args:
            csv_path (str): Path to the CSV file containing agent configurations
        """
        try:
            csv_path = self.load_path
            logger.info(
                f"Loading agent configurations from {csv_path}"
            )

            async with aiofiles.open(csv_path, mode="r") as file:
                content = await file.read()
                csv_reader = csv.DictReader(content.splitlines())

                for row in csv_reader:
                    config = AgentConfig(
                        agent_name=row["agent_name"],
                        description=row["description"],
                        system_prompt=row["system_prompt"],
                        task=row["task"],
                    )

                    # Create new agent with configuration
                    new_agent = Agent(
                        agent_name=config.agent_name,
                        system_prompt=config.system_prompt,
                        description=config.description,
                        model_name=(
                            row["model_name"]
                            if "model_name" in row
                            else "openai/gpt-4o"
                        ),
                        docs=[row["docs"]] if "docs" in row else "",
                        dynamic_temperature_enabled=True,
                        max_loops=(
                            row["max_loops"]
                            if "max_loops" in row
                            else 1
                        ),
                        user_name=(
                            row["user_name"]
                            if "user_name" in row
                            else "user"
                        ),
                        # output_type="str",
                        stopping_token=(
                            row["stopping_token"]
                            if "stopping_token" in row
                            else None
                        ),
                    )

                    # Add agent to swarm
                    self.agents.append(new_agent)
                    self.agent_configs[config.agent_name] = config

            # Update metadata with new agents
            self.metadata.agents = [
                agent.name for agent in self.agents
            ]
            self.metadata.number_of_agents = len(self.agents)
            logger.info(
                f"Loaded {len(self.agent_configs)} agent configurations"
            )
        except Exception as e:
            logger.error(f"Error loading agent configurations: {e}")

    def load_from_csv(self):
        asyncio.run(self._load_from_csv())

    async def run_from_config(self):
        """
        Run all agents with their configured tasks concurrently
        """
        logger.info("Running agents from configuration")
        self.metadata.start_time = time

        tasks = []
        for agent in self.agents:
            config = self.agent_configs.get(agent.agent_name)
            if config:
                for _ in range(self.max_loops):
                    tasks.append(
                        asyncio.to_thread(
                            self._run_agent_task, agent, config.task
                        )
                    )

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Process the results
        for result in results:
            self._track_output(*result)

        self.metadata.end_time = time

        # Save metadata
        logger.info("Saving metadata to CSV and JSON...")
        await self._save_metadata()

        if self.autosave_on:
            self.data_to_json_file()

        log_agent_data(self.metadata.model_dump())
        return self.metadata.model_dump_json(indent=4)

    async def _run(self, task: str = None, *args, **kwargs):
        """
        Run the swarm either with a specific task or using configured tasks.

        Args:
            task (str, optional): The task to be executed by all agents. If None, uses tasks from config.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The JSON representation of the swarm metadata.
        """
        if task is None and self.agent_configs:
            return await self.run_from_config()
        else:
            self.metadata.start_time = time
            await self._run_tasks(task, *args, **kwargs)
            self.metadata.end_time = time
            await self._save_metadata()

            if self.autosave_on:
                self.data_to_json_file()

            print(log_agent_data(self.metadata.model_dump()))
            return self.metadata.model_dump_json(indent=4)

    def run(self, task: str = None, *args, **kwargs):
        """
        Run the swarm with the specified task.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The JSON representation of the swarm metadata.

        """
        try:
            return asyncio.run(self._run(task, *args, **kwargs))
        except Exception as e:
            logger.error(f"Error running swarm: {e}")
            raise e

    async def _run_tasks(self, task: str, *args, **kwargs):
        """
        Run the swarm tasks concurrently.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        tasks = []
        for _ in range(self.max_loops):
            for agent in self.agents:
                # Use asyncio.to_thread to run the blocking task in a thread pool
                tasks.append(
                    asyncio.to_thread(
                        self._run_agent_task,
                        agent,
                        task,
                        *args,
                        **kwargs,
                    )
                )

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Process the results
        for result in results:
            self._track_output(*result)

    def _run_agent_task(self, agent, task, *args, **kwargs):
        """
        Run a single agent's task in a separate thread.

        Args:
            agent: The agent to run the task for.
            task (str): The task to be executed by the agent.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[str, str, str]: A tuple containing the agent name, task, and result.
        """
        try:
            result = agent.run(task=task, *args, **kwargs)
            # Assuming agent.run() is a blocking call
            return agent.agent_name, task, result
        except Exception as e:
            logger.error(
                f"Error running task for {agent.agent_name}: {e}"
            )
            return agent.agent_name, task, str(e)

    def _track_output(self, agent_name: str, task: str, result: str):
        """
        Track the output of a completed task.

        Args:
            agent_name (str): The name of the agent that completed the task.
            task (str): The task that was completed.
            result (str): The result of the completed task.
        """
        self.metadata.tasks_completed += 1
        self.metadata.outputs.append(
            AgentOutput(
                agent_name=agent_name,
                task=task,
                result=result,
                timestamp=time,
            )
        )

    def export_to_json(self):
        """
        Export the swarm metadata to JSON.

        Returns:
            str: The JSON representation of the swarm metadata.
        """
        return self.metadata.model_dump_json(indent=4)

    def data_to_json_file(self):
        """
        Save the swarm metadata to a JSON file.
        """
        out = self.export_to_json()

        create_file_in_folder(
            folder_path=f"{self.workspace_dir}/Spreedsheet-Swarm-{self.name}/{self.name}",
            file_name=f"spreedsheet-swarm-{uuid_hex}-metadata.json",
            content=out,
        )

    async def _save_metadata(self):
        """
        Save the swarm metadata to CSV and JSON.
        """
        if self.autosave_on:
            await self._save_to_csv()

    async def _save_to_csv(self):
        """
        Save the swarm metadata to a CSV file.
        """
        logger.info(
            f"Saving swarm metadata to: {self.save_file_path}"
        )
        run_id = uuid.uuid4()

        # Check if file exists before opening it
        file_exists = os.path.exists(self.save_file_path)

        async with aiofiles.open(
            self.save_file_path, mode="a"
        ) as file:
            writer = csv.writer(file)

            # Write header if file doesn't exist
            if not file_exists:
                await writer.writerow(
                    [
                        "Run ID",
                        "Agent Name",
                        "Task",
                        "Result",
                        "Timestamp",
                    ]
                )

            for output in self.metadata.outputs:
                await writer.writerow(
                    [
                        str(run_id),
                        output.agent_name,
                        output.task,
                        output.result,
                        output.timestamp,
                    ]
                )
