from typing import (
    List,
    Dict,
    TypedDict,
    Any,
    Union,
    TypeVar,
)
from dataclasses import dataclass
import csv
import json
import yaml
from pathlib import Path
from enum import Enum
from swarms.structs.agent import Agent
from swarms.schemas.swarms_api_schemas import AgentSpec
from litellm import model_list
import concurrent.futures
from tqdm import tqdm

# Type variable for agent configuration
AgentConfigType = TypeVar(
    "AgentConfigType", bound=Union[AgentSpec, Dict[str, Any]]
)


class ModelName(str, Enum):
    """Valid model names for swarms agents"""

    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
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
            if not any(
                model_name in model["model_name"]
                for model in model_list
            ):
                [model["model_name"] for model in model_list]
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


class AgentLoader:
    """Class to manage agents through various file formats with type safety and high performance"""

    def __init__(
        self, file_path: Union[str, Path], max_workers: int = 10
    ):
        """Initialize the AgentLoader with file path and max workers for parallel processing"""
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
