from swarms.utils.loguru_logger import logger
import yaml
from pydantic import BaseModel
from typing import List, Optional
import json
from swarms.structs.agent_registry import AgentRegistry
from swarms.structs.agent import Agent
from swarms.models.popular_llms import OpenAIChat


class AgentInput(BaseModel):
    agent_name: str = "Swarm Agent"
    system_prompt: Optional[str] = None
    agent_description: Optional[str] = None
    model_name: str = "OpenAIChat"
    max_loops: int = 1
    autosave: bool = False
    dynamic_temperature_enabled: bool = False
    dashboard: bool = False
    verbose: bool = False
    streaming_on: bool = True
    saved_state_path: Optional[str] = None
    sop: Optional[str] = None
    sop_list: Optional[List[str]] = None
    user_name: str = "User"
    retry_attempts: int = 3
    context_length: int = 8192
    task: Optional[str] = None
    interactive: bool = False


def parse_yaml_to_json(yaml_str: str) -> str:
    """
    Parses the given YAML string into an AgentInput model and converts it to a JSON string.

    Args:
        yaml_str (str): The YAML string to be parsed.

    Returns:
        str: The JSON string representation of the parsed YAML.

    Raises:
        ValueError: If the YAML string cannot be parsed into the AgentInput model.
    """
    try:
        data = yaml.safe_load(yaml_str)
        agent_input = AgentInput(**data)
        return agent_input.json()
    except yaml.YAMLError as e:
        print(f"YAML Error: {e}")
        raise ValueError("Invalid YAML input.") from e
    except ValueError as e:
        print(f"Validation Error: {e}")
        raise ValueError("Invalid data for AgentInput model.") from e


# # Example usage
# yaml_input = """
# agent_name: "Custom Agent"
# system_prompt: "System prompt example"
# agent_description: "This is a test agent"
# model_name: "CustomModel"
# max_loops: 5
# autosave: true
# dynamic_temperature_enabled: true
# dashboard: true
# verbose: true
# streaming_on: false
# saved_state_path: "/path/to/state"
# sop: "Standard operating procedure"
# sop_list: ["step1", "step2"]
# user_name: "Tester"
# retry_attempts: 5
# context_length: 4096
# task: "Perform testing"
# """

# json_output = parse_yaml_to_json(yaml_input)
# print(json_output)

registry = AgentRegistry()


def create_agent_from_yaml(yaml_path: str) -> None:
    with open(yaml_path, "r") as file:
        yaml_str = file.read()
    agent_json = parse_yaml_to_json(yaml_str)
    agent_config = json.loads(agent_json)

    agent = Agent(
        agent_name=agent_config.get("agent_name", "Swarm Agent"),
        system_prompt=agent_config.get("system_prompt"),
        agent_description=agent_config.get("agent_description"),
        llm=OpenAIChat(),
        max_loops=agent_config.get("max_loops", 1),
        autosave=agent_config.get("autosave", False),
        dynamic_temperature_enabled=agent_config.get(
            "dynamic_temperature_enabled", False
        ),
        dashboard=agent_config.get("dashboard", False),
        verbose=agent_config.get("verbose", False),
        streaming_on=agent_config.get("streaming_on", True),
        saved_state_path=agent_config.get("saved_state_path"),
        retry_attempts=agent_config.get("retry_attempts", 3),
        context_length=agent_config.get("context_length", 8192),
    )

    registry.add(agent.agent_name, agent)
    logger.info(f"Agent {agent.agent_name} created from {yaml_path}.")


def run_agent(agent_name: str, task: str) -> None:
    agent = registry.find_agent_by_name(agent_name)
    agent.run(task)


def list_agents() -> None:
    agents = registry.list_agents()
    for agent_id in agents:
        print(agent_id)
