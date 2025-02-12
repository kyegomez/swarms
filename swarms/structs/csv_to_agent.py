from typing import (
    List,
    Dict,
    TypedDict,
    Any,
)
from dataclasses import dataclass
import csv
from pathlib import Path
from enum import Enum
from swarms.structs.agent import Agent


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


class AgentConfigDict(TypedDict):
    """TypedDict for agent configuration"""

    agent_name: str
    system_prompt: str
    model_name: str  # Using str instead of ModelName for flexibility
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
    def validate_config(config: Dict[str, Any]) -> AgentConfigDict:
        """Validate and convert agent configuration"""
        try:
            # Validate model name
            model_name = str(config["model_name"])
            if not ModelName.is_valid_model(model_name):
                valid_models = ModelName.get_model_names()
                raise AgentValidationError(
                    f"Invalid model name. Must be one of: {', '.join(valid_models)}",
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
    """Class to manage agents through CSV with type safety"""

    csv_path: Path

    def __post_init__(self) -> None:
        """Convert string path to Path object if necessary"""
        if isinstance(self.csv_path, str):
            self.csv_path = Path(self.csv_path)

    @property
    def headers(self) -> List[str]:
        """CSV headers for agent configuration"""
        return [
            "agent_name",
            "system_prompt",
            "model_name",
            "max_loops",
            "autosave",
            "dashboard",
            "verbose",
            "dynamic_temperature",
            "saved_state_path",
            "user_name",
            "retry_attempts",
            "context_length",
            "return_step_meta",
            "output_type",
            "streaming",
        ]

    def create_agent_csv(self, agents: List[Dict[str, Any]]) -> None:
        """Create a CSV file with validated agent configurations"""
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

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(validated_agents)

        print(
            f"Created CSV with {len(validated_agents)} agents at {self.csv_path}"
        )

    def load_agents(self, file_type: str = "csv") -> List[Agent]:
        """Load and create agents from CSV or JSON with validation"""
        if file_type == "csv":
            if not self.csv_path.exists():
                raise FileNotFoundError(
                    f"CSV file not found at {self.csv_path}"
                )
            return self._load_agents_from_csv()
        elif file_type == "json":
            return self._load_agents_from_json()
        else:
            raise ValueError(
                "Unsupported file type. Use 'csv' or 'json'."
            )

    def _load_agents_from_csv(self) -> List[Agent]:
        """Load agents from a CSV file"""
        agents: List[Agent] = []
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    validated_config = AgentValidator.validate_config(
                        row
                    )
                    agent = self._create_agent(validated_config)
                    agents.append(agent)
                except AgentValidationError as e:
                    print(
                        f"Skipping invalid agent configuration: {e}"
                    )
                    continue

        print(f"Loaded {len(agents)} agents from {self.csv_path}")
        return agents

    def _load_agents_from_json(self) -> List[Agent]:
        """Load agents from a JSON file"""
        import json

        if not self.csv_path.with_suffix(".json").exists():
            raise FileNotFoundError(
                f"JSON file not found at {self.csv_path.with_suffix('.json')}"
            )

        agents: List[Agent] = []
        with open(self.csv_path.with_suffix(".json"), "r") as f:
            agents_data = json.load(f)
            for agent in agents_data:
                try:
                    validated_config = AgentValidator.validate_config(
                        agent
                    )
                    agent = self._create_agent(validated_config)
                    agents.append(agent)
                except AgentValidationError as e:
                    print(
                        f"Skipping invalid agent configuration: {e}"
                    )
                    continue

        print(
            f"Loaded {len(agents)} agents from {self.csv_path.with_suffix('.json')}"
        )
        return agents

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
