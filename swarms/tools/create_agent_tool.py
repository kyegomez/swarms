from typing import Union
from swarms.structs.agent import Agent
from swarms.schemas.agent_class_schema import AgentConfiguration
from functools import lru_cache
import json
from pydantic import ValidationError


def validate_and_convert_config(
    agent_configuration: Union[AgentConfiguration, dict, str],
) -> AgentConfiguration:
    """
    Validate and convert various input types to AgentConfiguration.

    Args:
        agent_configuration: Can be:
            - AgentConfiguration instance (BaseModel)
            - Dictionary with configuration parameters
            - JSON string representation of configuration

    Returns:
        AgentConfiguration: Validated configuration object

    Raises:
        ValueError: If input cannot be converted to valid AgentConfiguration
        ValidationError: If validation fails
    """
    if agent_configuration is None:
        raise ValueError("Agent configuration is required")

    # If already an AgentConfiguration instance, return as-is
    if isinstance(agent_configuration, AgentConfiguration):
        return agent_configuration

    # If string, try to parse as JSON
    if isinstance(agent_configuration, str):
        try:
            config_dict = json.loads(agent_configuration)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON string for agent configuration: {e}"
            )

        if not isinstance(config_dict, dict):
            raise ValueError(
                "JSON string must represent a dictionary/object"
            )

        agent_configuration = config_dict

    # If dictionary, convert to AgentConfiguration
    if isinstance(agent_configuration, dict):
        try:
            return AgentConfiguration(**agent_configuration)
        except ValidationError as e:
            raise ValueError(
                f"Invalid agent configuration parameters: {e}"
            )

    # If none of the above, raise error
    raise ValueError(
        f"agent_configuration must be AgentConfiguration instance, dict, or JSON string. "
        f"Got {type(agent_configuration)}"
    )


@lru_cache(maxsize=128)
def create_agent_tool(
    agent_configuration: Union[AgentConfiguration, dict, str],
) -> Agent:
    """
    Create an agent tool from an agent configuration.
    Uses caching to improve performance for repeated configurations.

    Args:
        agent_configuration: Agent configuration as:
            - AgentConfiguration instance (BaseModel)
            - Dictionary with configuration parameters
            - JSON string representation of configuration
        function: Agent class or function to create the agent

    Returns:
        Callable: Configured agent instance

    Raises:
        ValueError: If agent_configuration is invalid or cannot be converted
        ValidationError: If configuration validation fails
    """
    # Validate and convert configuration
    config = validate_and_convert_config(agent_configuration)

    agent = Agent(
        agent_name=config.agent_name,
        agent_description=config.agent_description,
        system_prompt=config.system_prompt,
        max_loops=config.max_loops,
        dynamic_temperature_enabled=config.dynamic_temperature_enabled,
        model_name=config.model_name,
        safety_prompt_on=config.safety_prompt_on,
        temperature=config.temperature,
        output_type="str-all-except-first",
    )

    return agent.run(task=config.task)
