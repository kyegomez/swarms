import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.structs.agent import Agent
from swarms.structs.swarm_router import SwarmRouter
from swarms.utils.litellm_wrapper import LiteLLM

logger = initialize_logger(log_folder="create_agents_from_yaml")


class AgentConfig(BaseModel):
    agent_name: str
    system_prompt: str
    model_name: Optional[str] = None
    max_loops: int = Field(default=1, ge=1)
    autosave: bool = True
    dashboard: bool = False
    verbose: bool = False
    dynamic_temperature_enabled: bool = False
    saved_state_path: Optional[str] = None
    user_name: str = "default_user"
    retry_attempts: int = Field(default=3, ge=1)
    context_length: int = Field(default=100000, ge=1000)
    return_step_meta: bool = False
    output_type: str = "str"
    auto_generate_prompt: bool = False
    artifacts_on: bool = False
    artifacts_file_extension: str = ".md"
    artifacts_output_path: str = ""

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError(
                "System prompt must be a non-empty string"
            )
        return v


class SwarmConfig(BaseModel):
    name: str
    description: str
    max_loops: int = Field(default=1, ge=1)
    swarm_type: str
    task: Optional[str] = None
    flow: Optional[Dict] = None
    autosave: bool = True
    return_json: bool = False
    rules: str = ""

    @field_validator("swarm_type")
    @classmethod
    def validate_swarm_type(cls, v):
        valid_types = {
            "SequentialWorkflow",
            "ConcurrentWorkflow",
            "AgentRearrange",
            "MixtureOfAgents",
            "auto",
        }
        if v not in valid_types:
            raise ValueError(
                f"Swarm type must be one of: {valid_types}"
            )
        return v


class YAMLConfig(BaseModel):
    agents: List[AgentConfig] = Field(..., min_length=1)
    swarm_architecture: Optional[SwarmConfig] = None

    model_config = {
        "extra": "forbid"  # Prevent additional fields not in the model
    }


def load_yaml_safely(
    yaml_file: str = None, yaml_string: str = None
) -> Dict:
    """Safely load and validate YAML configuration using Pydantic."""
    try:
        if yaml_string:
            config_dict = yaml.safe_load(yaml_string)
        elif yaml_file:
            if not os.path.exists(yaml_file):
                raise FileNotFoundError(
                    f"YAML file {yaml_file} not found."
                )
            with open(yaml_file, "r") as file:
                config_dict = yaml.safe_load(file)
        else:
            raise ValueError(
                "Either yaml_file or yaml_string must be provided"
            )

        # Validate using Pydantic
        YAMLConfig(**config_dict)
        return config_dict
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error validating configuration: {str(e)}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying after error: {retry_state.outcome.exception()}"
    ),
)
def create_agent_with_retry(
    agent_config: Dict, model: LiteLLM
) -> Agent:
    """Create an agent with retry logic for handling transient failures."""
    try:
        validated_config = AgentConfig(**agent_config)
        agent = Agent(
            agent_name=validated_config.agent_name,
            system_prompt=validated_config.system_prompt,
            llm=model,
            max_loops=validated_config.max_loops,
            autosave=validated_config.autosave,
            dashboard=validated_config.dashboard,
            verbose=validated_config.verbose,
            dynamic_temperature_enabled=validated_config.dynamic_temperature_enabled,
            saved_state_path=validated_config.saved_state_path,
            user_name=validated_config.user_name,
            retry_attempts=validated_config.retry_attempts,
            context_length=validated_config.context_length,
            return_step_meta=validated_config.return_step_meta,
            output_type=validated_config.output_type,
            auto_generate_prompt=validated_config.auto_generate_prompt,
            artifacts_on=validated_config.artifacts_on,
            artifacts_file_extension=validated_config.artifacts_file_extension,
            artifacts_output_path=validated_config.artifacts_output_path,
        )
        return agent
    except Exception as e:
        logger.error(
            f"Error creating agent {agent_config.get('agent_name', 'unknown')}: {str(e)}"
        )
        raise


def create_agents_from_yaml(
    model: Callable = None,
    yaml_file: str = "agents.yaml",
    yaml_string: str = None,
    return_type: str = "auto",
) -> Union[
    SwarmRouter,
    Agent,
    List[Agent],
    Tuple[Union[SwarmRouter, Agent], List[Agent]],
    List[Dict[str, Any]],
]:
    """
    Create agents and/or SwarmRouter based on configurations defined in a YAML file or string.
    """
    agents = []
    task_results = []
    swarm_router = None

    try:
        logger.info("Starting agent creation process...")

        # Load and validate configuration
        if yaml_file:
            logger.info(f"Loading configuration from {yaml_file}")
        config = load_yaml_safely(yaml_file, yaml_string)

        if not config.get("agents"):
            raise ValueError(
                "No agents defined in the YAML configuration. "
                "Please add at least one agent under the 'agents' section."
            )

        logger.info(
            f"Found {len(config['agents'])} agent(s) to create"
        )

        # Create agents with retry logic
        for idx, agent_config in enumerate(config["agents"], 1):
            if not agent_config.get("agent_name"):
                agent_config["agent_name"] = f"Agent_{idx}"

            logger.info(
                f"Creating agent {idx}/{len(config['agents'])}: {agent_config['agent_name']}"
            )

            if "model_name" in agent_config:
                logger.info(
                    f"Using specified model: {agent_config['model_name']}"
                )
                model_instance = LiteLLM(
                    model_name=agent_config["model_name"]
                )
            else:
                model_name = "gpt-4"
                logger.info(
                    f"No model specified, using default: {model_name}"
                )
                model_instance = LiteLLM(model_name=model_name)

            agent = create_agent_with_retry(
                agent_config, model_instance
            )
            logger.info(
                f"Agent {agent_config['agent_name']} created successfully."
            )
            agents.append(agent)

        logger.info(f"Successfully created {len(agents)} agent(s)")

        # Create SwarmRouter if specified
        if "swarm_architecture" in config:
            logger.info("Setting up swarm architecture...")
            try:
                if not isinstance(config["swarm_architecture"], dict):
                    raise ValueError(
                        "swarm_architecture must be a dictionary containing swarm configuration"
                    )

                required_fields = {
                    "name",
                    "description",
                    "swarm_type",
                }
                missing_fields = required_fields - set(
                    config["swarm_architecture"].keys()
                )
                if missing_fields:
                    raise ValueError(
                        f"SwarmRouter creation failed: Missing required fields in swarm_architecture: {', '.join(missing_fields)}"
                    )

                swarm_config = SwarmConfig(
                    **config["swarm_architecture"]
                )

                logger.info(
                    f"Creating SwarmRouter with type: {swarm_config.swarm_type}"
                )
                swarm_router = SwarmRouter(
                    name=swarm_config.name,
                    description=swarm_config.description,
                    max_loops=swarm_config.max_loops,
                    agents=agents,
                    swarm_type=swarm_config.swarm_type,
                    task=swarm_config.task,
                    flow=swarm_config.flow,
                    autosave=swarm_config.autosave,
                    return_json=swarm_config.return_json,
                    rules=swarm_config.rules,
                )
                logger.info(
                    f"SwarmRouter '{swarm_config.name}' created successfully."
                )
            except Exception as e:
                logger.error(f"Error creating SwarmRouter: {str(e)}")
                if "swarm_type" in str(e) and "valid_types" in str(e):
                    raise ValueError(
                        "Invalid swarm_type. Must be one of: SequentialWorkflow, ConcurrentWorkflow, "
                        "AgentRearrange, MixtureOfAgents, or auto"
                    )
                raise ValueError(
                    f"Failed to create SwarmRouter: {str(e)}. Make sure your YAML file "
                    "has a valid swarm_architecture section with required fields."
                )

        # Handle return types with improved error checking
        valid_return_types = {
            "auto",
            "swarm",
            "agents",
            "both",
            "tasks",
            "run_swarm",
        }
        if return_type not in valid_return_types:
            raise ValueError(
                f"Invalid return_type. Must be one of: {valid_return_types}"
            )

        logger.info(f"Processing with return type: {return_type}")

        if return_type in ("run_swarm", "swarm"):
            if not swarm_router:
                if "swarm_architecture" not in config:
                    raise ValueError(
                        "Cannot run swarm: No swarm_architecture section found in YAML configuration.\n"
                        "Please add a swarm_architecture section with:\n"
                        "  - name: your_swarm_name\n"
                        "  - description: your_swarm_description\n"
                        "  - swarm_type: one of [SequentialWorkflow, ConcurrentWorkflow, AgentRearrange, MixtureOfAgents, auto]\n"
                        "  - task: your_task_description"
                    )
                raise ValueError(
                    "Cannot run swarm: SwarmRouter creation failed. Check the previous error messages."
                )
            try:
                if not config["swarm_architecture"].get("task"):
                    raise ValueError(
                        "No task specified in swarm_architecture. Please add a 'task' field "
                        "to define what the swarm should do."
                    )
                logger.info(
                    f"Running swarm with task: {config['swarm_architecture']['task']}"
                )
                return swarm_router.run(
                    config["swarm_architecture"]["task"]
                )
            except Exception as e:
                logger.error(f"Error running SwarmRouter: {str(e)}")
                raise

        # Return appropriate type based on configuration
        if return_type == "auto":
            result = (
                swarm_router
                if swarm_router
                else (agents[0] if len(agents) == 1 else agents)
            )
        elif return_type == "swarm":
            result = (
                swarm_router
                if swarm_router
                else (agents[0] if len(agents) == 1 else agents)
            )
        elif return_type == "agents":
            result = agents[0] if len(agents) == 1 else agents
        elif return_type == "both":
            result = (
                (
                    swarm_router
                    if swarm_router
                    else agents[0] if len(agents) == 1 else agents
                ),
                agents,
            )
        elif return_type == "tasks":
            result = task_results

        logger.info("Process completed successfully")
        return result

    except Exception as e:
        logger.error(
            f"Critical error in create_agents_from_yaml: {str(e)}\n"
            "Please check your YAML configuration and try again."
        )
        raise
