import concurrent.futures
import csv
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    TypedDict,
    TypeVar,
    Union,
)

import yaml
from litellm import model_list
from tqdm import tqdm

from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)
from swarms.schemas.swarms_api_schemas import AgentSpec
from swarms.utils.types import ReturnTypes
from swarms.structs.agent import Agent
from swarms.utils.agent_loader_markdown import (
    load_agent_from_markdown,
    load_agents_from_markdown,
    MarkdownAgentLoader,
)

# Type variable for agent configuration
AgentConfigType = TypeVar(
    "AgentConfigType", bound=Union[AgentSpec, Dict[str, Any]]
)


class ModelName(str, Enum):
    """Valid model names for swarms agents"""

    GPT4O = "gpt-5.4"
    GPT4O_MINI = "gpt-5.4"
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE = "claude-v1"
    CLAUDE2 = "claude-2"

    @classmethod
    def get_model_names(cls) -> List[str]:
        """Get list of valid model names"""
        return [model.value for model in cls]

    @classmethod
    def is_valid_model(cls, model_name: str) -> bool:
        """Check if model name is valid"""
        return model_name in cls.get_model_names()


class FileType(str, Enum):
    """Supported file types for agent configuration"""

    CSV = "csv"
    JSON = "json"
    YAML = "yaml"


class AgentConfigDict(TypedDict):
    """TypedDict for agent configuration"""

    agent_name: str
    system_prompt: str
    model_name: str
    max_loops: int
    autosave: bool
    dashboard: bool
    verbose: bool
    dynamic_temperature: bool
    saved_state_path: str
    user_name: str
    retry_attempts: int
    context_length: int
    return_step_meta: bool
    output_type: str
    streaming: bool


@dataclass
class AgentValidationError(Exception):
    """Custom exception for agent validation errors"""

    message: str
    field: str
    value: Any

    def __str__(self) -> str:
        return f"Validation error in field '{self.field}': {self.message}. Got value: {self.value}"


class AgentValidator:
    """Validates agent configuration data"""

    @staticmethod
    def validate_config(
        config: Union[AgentSpec, Dict[str, Any]],
    ) -> AgentConfigDict:
        """Validate and convert agent configuration from either AgentSpec or Dict"""
        try:
            # Convert AgentSpec to dict if needed
            if isinstance(config, AgentSpec):
                config = config.model_dump()

            # Validate model name using litellm model list
            model_name = str(config["model_name"])
            # model_list from litellm is a list of strings, not dicts
            if isinstance(model_list, list) and len(model_list) > 0:
                if isinstance(model_list[0], str):
                    # model_list is list of strings
                    if not any(
                        model_name in model or model in model_name
                        for model in model_list
                    ):
                        raise AgentValidationError(
                            "Invalid model name. Must be one of the supported litellm models",
                            "model_name",
                            model_name,
                        )
                elif isinstance(model_list[0], dict):
                    # model_list is list of dicts (fallback for different litellm versions)
                    if not any(
                        model_name in model.get("model_name", "")
                        for model in model_list
                    ):
                        raise AgentValidationError(
                            "Invalid model name. Must be one of the supported litellm models",
                            "model_name",
                            model_name,
                        )

            # Convert types with error handling
            validated_config: AgentConfigDict = {
                "agent_name": str(config.get("agent_name", "")),
                "system_prompt": str(config.get("system_prompt", "")),
                "model_name": model_name,
                "max_loops": int(config.get("max_loops", 1)),
                "autosave": bool(
                    str(config.get("autosave", True)).lower()
                    == "true"
                ),
                "dashboard": bool(
                    str(config.get("dashboard", False)).lower()
                    == "true"
                ),
                "verbose": bool(
                    str(config.get("verbose", True)).lower() == "true"
                ),
                "dynamic_temperature": bool(
                    str(
                        config.get("dynamic_temperature", True)
                    ).lower()
                    == "true"
                ),
                "saved_state_path": str(
                    config.get("saved_state_path", "")
                ),
                "user_name": str(
                    config.get("user_name", "default_user")
                ),
                "retry_attempts": int(
                    config.get("retry_attempts", 3)
                ),
                "context_length": int(
                    config.get("context_length", 200000)
                ),
                "return_step_meta": bool(
                    str(config.get("return_step_meta", False)).lower()
                    == "true"
                ),
                "output_type": str(
                    config.get("output_type", "string")
                ),
                "streaming": bool(
                    str(config.get("streaming", False)).lower()
                    == "true"
                ),
            }

            return validated_config

        except (ValueError, KeyError) as e:
            raise AgentValidationError(
                str(e), str(e.__class__.__name__), str(config)
            )


class CSVAgentLoader:
    """Class to manage agents through various file formats with type safety and high performance"""

    def __init__(
        self, file_path: Union[str, Path], max_workers: int = 10
    ):
        """Initialize the CSVAgentLoader with file path and max workers for parallel processing"""
        self.file_path = (
            Path(file_path)
            if isinstance(file_path, str)
            else file_path
        )
        self.max_workers = max_workers

    @property
    def file_type(self) -> FileType:
        """Determine the file type based on extension"""
        ext = self.file_path.suffix.lower()
        if ext == ".csv":
            return FileType.CSV
        elif ext == ".json":
            return FileType.JSON
        elif ext in [".yaml", ".yml"]:
            return FileType.YAML
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def create_agent_file(
        self, agents: List[Union[AgentSpec, Dict[str, Any]]]
    ) -> None:
        """Create a file with validated agent configurations"""
        validated_agents = []
        for agent in agents:
            try:
                validated_config = AgentValidator.validate_config(
                    agent
                )
                validated_agents.append(validated_config)
            except AgentValidationError as e:
                print(
                    f"Validation error for agent {agent.get('agent_name', 'unknown')}: {e}"
                )
                raise

        if self.file_type == FileType.CSV:
            self._write_csv(validated_agents)
        elif self.file_type == FileType.JSON:
            self._write_json(validated_agents)
        elif self.file_type == FileType.YAML:
            self._write_yaml(validated_agents)

        print(
            f"Created {self.file_type.value} file with {len(validated_agents)} agents at {self.file_path}"
        )

    def load_agents(self) -> List[Agent]:
        """Load and create agents from file with validation and parallel processing"""
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found at {self.file_path}"
            )

        if self.file_type == FileType.CSV:
            agents_data = self._read_csv()
        elif self.file_type == FileType.JSON:
            agents_data = self._read_json()
        elif self.file_type == FileType.YAML:
            agents_data = self._read_yaml()

        # Process agents in parallel with progress bar
        agents: List[Agent] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = []
            for agent_data in agents_data:
                futures.append(
                    executor.submit(self._process_agent, agent_data)
                )

            # Use tqdm to show progress
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Loading agents",
            ):
                try:
                    agent = future.result()
                    if agent:
                        agents.append(agent)
                except Exception as e:
                    print(f"Error processing agent: {e}")

        print(f"Loaded {len(agents)} agents from {self.file_path}")
        return agents

    def _process_agent(
        self, agent_data: Union[AgentSpec, Dict[str, Any]]
    ) -> Union[Agent, None]:
        """Process a single agent configuration"""
        try:
            validated_config = AgentValidator.validate_config(
                agent_data
            )
            return self._create_agent(validated_config)
        except AgentValidationError as e:
            print(f"Skipping invalid agent configuration: {e}")
            return None

    def _create_agent(
        self, validated_config: AgentConfigDict
    ) -> Agent:
        """Create an Agent instance from validated configuration"""
        return Agent(
            agent_name=validated_config["agent_name"],
            system_prompt=validated_config["system_prompt"],
            model_name=validated_config["model_name"],
            max_loops=validated_config["max_loops"],
            autosave=validated_config["autosave"],
            dashboard=validated_config["dashboard"],
            verbose=validated_config["verbose"],
            dynamic_temperature_enabled=validated_config[
                "dynamic_temperature"
            ],
            saved_state_path=validated_config["saved_state_path"],
            user_name=validated_config["user_name"],
            retry_attempts=validated_config["retry_attempts"],
            context_length=validated_config["context_length"],
            return_step_meta=validated_config["return_step_meta"],
            output_type=validated_config["output_type"],
            streaming_on=validated_config["streaming"],
        )

    def _write_csv(self, agents: List[Dict[str, Any]]) -> None:
        """Write agents to CSV file"""
        with open(self.file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=agents[0].keys())
            writer.writeheader()
            writer.writerows(agents)

    def _write_json(self, agents: List[Dict[str, Any]]) -> None:
        """Write agents to JSON file"""
        with open(self.file_path, "w") as f:
            json.dump(agents, f, indent=2)

    def _write_yaml(self, agents: List[Dict[str, Any]]) -> None:
        """Write agents to YAML file"""
        with open(self.file_path, "w") as f:
            yaml.dump(agents, f, default_flow_style=False)

    def _read_csv(self) -> List[Dict[str, Any]]:
        """Read agents from CSV file"""
        with open(self.file_path, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _read_json(self) -> List[Dict[str, Any]]:
        """Read agents from JSON file"""
        with open(self.file_path, "r") as f:
            return json.load(f)

    def _read_yaml(self) -> List[Dict[str, Any]]:
        """Read agents from YAML file"""
        with open(self.file_path, "r") as f:
            return yaml.safe_load(f)


class AgentLoader:
    """
    Loader class for creating Agent objects from various file formats.

    This class provides methods to load agents from Markdown, YAML, and CSV files.
    """

    def __init__(self, concurrent: bool = True):
        """
        Initialize the AgentLoader instance.
        """
        self.concurrent = concurrent
        pass

    def load_agents_from_markdown(
        self,
        file_paths: Union[str, List[str]],
        concurrent: bool = True,
        max_file_size_mb: float = 10.0,
        **kwargs,
    ) -> List[Agent]:
        """
        Load multiple agents from one or more Markdown files.

        Args:
            file_paths (Union[str, List[str]]): Path or list of paths to Markdown file(s) containing agent definitions.
            concurrent (bool, optional): Whether to load files concurrently. Defaults to True.
            max_file_size_mb (float, optional): Maximum file size in MB to process. Defaults to 10.0.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects.
        """
        return load_agents_from_markdown(
            file_paths=file_paths,
            concurrent=concurrent,
            max_file_size_mb=max_file_size_mb,
            **kwargs,
        )

    def load_agent_from_markdown(
        self, file_path: str, **kwargs
    ) -> Agent:
        """
        Load a single agent from a Markdown file.

        Args:
            file_path (str): Path to the Markdown file containing the agent definition.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            Agent: The loaded Agent object.
        """
        return load_agent_from_markdown(file_path=file_path, **kwargs)

    def load_agents_from_yaml(
        self,
        yaml_file: str,
        return_type: ReturnTypes = "auto",
        **kwargs,
    ) -> List[Agent]:
        """
        Load agents from a YAML file.

        Args:
            yaml_file (str): Path to the YAML file containing agent definitions.
            return_type (ReturnTypes, optional): The return type for the loader. Defaults to "auto".
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects.
        """
        return create_agents_from_yaml(
            yaml_file=yaml_file, return_type=return_type, **kwargs
        )

    def load_many_agents_from_yaml(
        self,
        yaml_files: List[str],
        return_types: List[ReturnTypes] = ["auto"],
        **kwargs,
    ) -> List[Agent]:
        """
        Load agents from multiple YAML files.

        Args:
            yaml_files (List[str]): List of YAML file paths containing agent definitions.
            return_types (List[ReturnTypes], optional): List of return types for each YAML file. Defaults to ["auto"].
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects from all files.
        """
        return [
            self.load_agents_from_yaml(
                yaml_file=yaml_file,
                return_type=return_types[i],
                **kwargs,
            )
            for i, yaml_file in enumerate(yaml_files)
        ]

    def load_agents_from_csv(
        self, csv_file: str, **kwargs
    ) -> List[Agent]:
        """
        Load agents from a CSV file.

        Args:
            csv_file (str): Path to the CSV file containing agent definitions.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects.
        """
        loader = CSVAgentLoader(file_path=csv_file)
        return loader.load_agents()

    def auto(self, file_path: str, *args, **kwargs):
        """
        Automatically load agents from a file based on its extension.

        Args:
            file_path (str): Path to the agent file (Markdown, YAML, or CSV).
            *args: Additional positional arguments passed to the underlying loader.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects.

        Raises:
            ValueError: If the file type is not supported.
        """
        if file_path.endswith(".md"):
            return self.load_agents_from_markdown(
                file_path, *args, **kwargs
            )
        elif file_path.endswith(".yaml"):
            return self.load_agents_from_yaml(
                file_path, *args, **kwargs
            )
        elif file_path.endswith(".csv"):
            return self.load_agents_from_csv(
                file_path, *args, **kwargs
            )
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def load_single_agent(self, *args, **kwargs):
        """
        Load a single agent from a file of a supported type.

        Args:
            *args: Positional arguments passed to the underlying loader.
            **kwargs: Keyword arguments passed to the underlying loader.

        Returns:
            Agent: The loaded Agent object.
        """
        return self.auto(*args, **kwargs)

    def load_multiple_agents(
        self, file_paths: List[str], *args, **kwargs
    ):
        """
        Load multiple agents from a list of files of various supported types.

        Args:
            file_paths (List[str]): List of file paths to agent files (Markdown, YAML, or CSV).
            *args: Additional positional arguments passed to the underlying loader.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects from all files.
        """
        return [
            self.auto(file_path, *args, **kwargs)
            for file_path in file_paths
        ]

    def parse_markdown_file(self, file_path: str):
        """
        Parse a Markdown file and return the agents defined within.

        Args:
            file_path (str): Path to the Markdown file.

        Returns:
            List[Agent]: A list of Agent objects parsed from the file.
        """
        return MarkdownAgentLoader(
            max_workers=os.cpu_count()
        ).parse_markdown_file(file_path=file_path)
