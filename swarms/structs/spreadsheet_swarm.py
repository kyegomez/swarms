import csv
import datetime
import os
import uuid
from typing import List

from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import (
    run_agents_with_different_tasks,
)
from swarms.structs.omni_agent_types import AgentType
from swarms.utils.file_processing import create_file_in_folder
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="spreadsheet_swarm")

time = datetime.datetime.now().isoformat()
uuid_hex = uuid.uuid4().hex

# --------------- NEW CHANGE START ---------------
# Format time variable to be compatible across operating systems
formatted_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# --------------- NEW CHANGE END ---------------


class SpreadSheetSwarm:
    """
    A swarm that processes tasks concurrently using multiple agents.

    Args:
        name (str, optional): The name of the swarm. Defaults to "Spreadsheet-Swarm".
        description (str, optional): The description of the swarm. Defaults to "A swarm that processes tasks concurrently using multiple agents.".
        agents (Union[Agent, List[Agent], None], optional): The agents participating in the swarm. If None, agents will be loaded from load_path. Defaults to None.
        autosave (bool, optional): Whether to enable autosave of swarm metadata. Defaults to True.
        save_file_path (str, optional): The file path to save the swarm metadata as a CSV file. Defaults to "spreedsheet_swarm.csv".
        max_loops (int, optional): The number of times to repeat the swarm tasks. Defaults to 1.
        workspace_dir (str, optional): The directory path of the workspace. Defaults to the value of the "WORKSPACE_DIR" environment variable.
        load_path (str, optional): Path to CSV file containing agent configurations. Required if agents is None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Note:
        Either 'agents' or 'load_path' must be provided. If both are provided, 'agents' will be used.
    """

    def __init__(
        self,
        name: str = "Spreadsheet-Swarm",
        description: str = "A swarm that processes tasks concurrently using multiple agents and saves the metadata to a CSV file.",
        agents: List[AgentType] = None,
        autosave: bool = True,
        save_file_path: str = None,
        max_loops: int = 1,
        workspace_dir: str = os.getenv("WORKSPACE_DIR"),
        load_path: str = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.save_file_path = save_file_path
        self.autosave = autosave
        self.max_loops = max_loops
        self.workspace_dir = workspace_dir
        self.load_path = load_path
        self.verbose = verbose

        # --------------- NEW CHANGE START ---------------
        # The save_file_path now uses the formatted_time and uuid_hex
        # Save CSV files in the workspace_dir instead of root directory
        if self.workspace_dir:
            os.makedirs(self.workspace_dir, exist_ok=True)
            self.save_file_path = os.path.join(
                self.workspace_dir,
                f"spreadsheet_swarm_run_id_{uuid_hex}.csv",
            )
        else:
            self.save_file_path = (
                f"spreadsheet_swarm_run_id_{uuid_hex}.csv"
            )
        # --------------- NEW CHANGE END ---------------

        self.outputs = []
        self.tasks_completed = 0
        self.agent_tasks = {}  # Simple dict to store agent tasks

        self.reliability_check()

    def reliability_check(self):
        """
        Check the reliability of the swarm.

        Raises:
            ValueError: If neither agents nor load_path is provided, or if max_loops is not provided.
        """
        if self.verbose:
            logger.info(
                f"SpreadSheetSwarm Name: {self.name} reliability checks in progress..."
            )

        if not self.agents:
            raise ValueError("No agents are provided.")

        if not self.max_loops:
            raise ValueError("No max loops are provided.")

        if self.verbose:
            logger.info(
                f"SpreadSheetSwarm Name: {self.name} reliability check passed."
            )
            logger.info(
                f"SpreadSheetSwarm Name: {self.name} is ready to run."
            )

    def _load_from_csv(self):
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

            with open(csv_path, mode="r") as file:
                csv_reader = csv.DictReader(file)

                for row in csv_reader:
                    agent_name = row["agent_name"]
                    task = row["task"]

                    # Store agent task mapping
                    self.agent_tasks[agent_name] = task

                    # Create new agent with configuration
                    new_agent = Agent(
                        agent_name=agent_name,
                        system_prompt=row["system_prompt"],
                        description=row["description"],
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
                        stopping_token=(
                            row["stopping_token"]
                            if "stopping_token" in row
                            else None
                        ),
                    )

                    # Add agent to swarm
                    self.agents.append(new_agent)

            # Agents have been loaded successfully
            logger.info(
                f"Loaded {len(self.agents)} agent configurations"
            )
        except Exception as e:
            logger.error(f"Error loading agent configurations: {e}")

    def load_from_csv(self):
        self._load_from_csv()

    def run_from_config(self):
        """
        Run all agents with their configured tasks concurrently
        """
        logger.info("Running agents from configuration")

        # Load agents from CSV if no agents are provided but load_path is
        if not self.agents and self.load_path:
            self.load_from_csv()

        start_time = time

        # Prepare agent-task pairs for concurrent execution
        agent_task_pairs = []

        for agent in self.agents:
            task = self.agent_tasks.get(agent.agent_name)
            if task:
                for _ in range(self.max_loops):
                    agent_task_pairs.append((agent, task))

        # Run all tasks concurrently using the multi_agent_exec function
        results = run_agents_with_different_tasks(agent_task_pairs)

        # Process the results
        for i, result in enumerate(results):
            agent, task = agent_task_pairs[i]
            self._track_output(agent.agent_name, task, result)

        end_time = time

        # Save outputs
        logger.info("Saving outputs to CSV...")
        self._save_metadata()

        if self.autosave:
            self.data_to_json_file()

        # Return simple summary
        return {
            "run_id": f"spreadsheet_swarm_run_{uuid_hex}",
            "name": self.name,
            "description": self.description,
            "start_time": start_time,
            "end_time": end_time,
            "tasks_completed": self.tasks_completed,
            "number_of_agents": len(self.agents),
            "outputs": self.outputs,
        }

    def _run(self, task: str = None, *args, **kwargs):
        """
        Run the swarm either with a specific task or using configured tasks.

        Args:
            task (str, optional): The task to be executed by all agents. If None, uses tasks from config.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Summary of the swarm execution.
        """
        # Load agents from CSV if no agents are provided but load_path is
        if not self.agents and self.load_path:
            self.load_from_csv()

        if task is None and self.agent_tasks:
            return self.run_from_config()
        else:
            start_time = time
            self._run_tasks(task, *args, **kwargs)
            end_time = time
            self._save_metadata()

            if self.autosave:
                self.data_to_json_file()

            # Return simple summary
            return {
                "run_id": f"spreadsheet_swarm_run_{uuid_hex}",
                "name": self.name,
                "description": self.description,
                "start_time": start_time,
                "end_time": end_time,
                "tasks_completed": self.tasks_completed,
                "number_of_agents": len(self.agents),
                "outputs": self.outputs,
            }

    def run(self, task: str = None, *args, **kwargs):
        """
        Run the swarm with the specified task.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Summary of the swarm execution.

        """
        try:
            return self._run(task, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error running swarm: {e}")
            raise e

    def _run_tasks(self, task: str, *args, **kwargs):
        """
        Run the swarm tasks concurrently.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        # Load agents from CSV if no agents are provided but load_path is
        if not self.agents and self.load_path:
            self.load_from_csv()

        # Prepare agents and tasks for concurrent execution
        agents_to_run = []
        tasks_to_run = []

        for _ in range(self.max_loops):
            for agent in self.agents:
                agents_to_run.append(agent)
                tasks_to_run.append(task)

        # Run all tasks concurrently using the multi_agent_exec function
        results = run_agents_with_different_tasks(
            list(zip(agents_to_run, tasks_to_run))
        )

        # Process the results
        for i, result in enumerate(results):
            agent = agents_to_run[i]
            task_str = tasks_to_run[i]
            self._track_output(agent.agent_name, task_str, result)

    def _track_output(self, agent_name: str, task: str, result: str):
        """
        Track the output of a completed task.

        Args:
            agent_name (str): The name of the agent that completed the task.
            task (str): The task that was completed.
            result (str): The result of the completed task.
        """
        self.tasks_completed += 1
        self.outputs.append(
            {
                "agent_name": agent_name,
                "task": task,
                "result": result,
                "timestamp": time,
            }
        )

    def export_to_json(self):
        """
        Export the swarm outputs to JSON.

        Returns:
            str: The JSON representation of the swarm outputs.
        """
        import json

        return json.dumps(
            {
                "run_id": f"spreadsheet_swarm_run_{uuid_hex}",
                "name": self.name,
                "description": self.description,
                "tasks_completed": self.tasks_completed,
                "number_of_agents": len(self.agents),
                "outputs": self.outputs,
            },
            indent=4,
        )

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

    def _save_metadata(self):
        """
        Save the swarm metadata to CSV and JSON.
        """
        if self.autosave:
            self._save_to_csv()

    def _save_to_csv(self):
        """
        Save the swarm metadata to a CSV file.
        """
        logger.info(
            f"Saving swarm metadata to: {self.save_file_path}"
        )
        run_id = uuid.uuid4()

        # Check if file exists before opening it
        file_exists = os.path.exists(self.save_file_path)

        with open(self.save_file_path, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow(
                    [
                        "Run ID",
                        "Agent Name",
                        "Task",
                        "Result",
                        "Timestamp",
                    ]
                )

            for output in self.outputs:
                writer.writerow(
                    [
                        str(run_id),
                        output["agent_name"],
                        output["task"],
                        output["result"],
                        output["timestamp"],
                    ]
                )
