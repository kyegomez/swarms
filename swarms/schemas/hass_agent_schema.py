from swarms.utils.loguru_logger import logger
import re
import json
from pydantic import BaseModel, Field
from typing import List
from swarms.structs.agent import Agent


class HaSAgentSchema(BaseModel):
    name: str = Field(
        ...,
        title="Name of the agent",
        description="Name of the agent",
    )
    system_prompt: str = (
        Field(
            ...,
            title="System prompt for the agent",
            description="System prompt for the agent",
        ),
    )
    rules: str = Field(
        ...,
        title="Rules",
        description="Rules for the agent",
    )


class HassSchema(BaseModel):
    agents: List[HaSAgentSchema] = Field(
        ...,
        title="List of agents to use for the problem",
        description="List of agents to use for the problem",
    )


# import json
def parse_json_from_input(input_str: str = None):
    """
    Parses a JSON string from the input and returns the parsed data.

    Args:
        input_str (str): The input string containing the JSON.

    Returns:
        tuple: A tuple containing the parsed data. The tuple contains three elements:
            - The plan extracted from the JSON.
            - The agents extracted from the JSON.
            - The rules extracted from the JSON.

            If the input string is None or empty, or if the JSON decoding fails, all elements of the tuple will be None.
    """
    # Validate input is not None or empty
    if not input_str:
        logger.info("Error: Input string is None or empty.")
        return None, None, None

    # Attempt to extract JSON from markdown using regular expression
    json_pattern = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
    match = json_pattern.search(input_str)
    json_str = match.group(1).strip() if match else input_str.strip()

    # Attempt to parse the JSON string
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.info(f"Error: JSON decoding failed with message '{e}'")
        return None, None, None

    hass_schema = HassSchema(**data)
    return (hass_schema.agents,)


## [Create the agents]
def create_worker_agents(
    agents: List[HassSchema],
    *args,
    **kwargs,
) -> List[Agent]:
    """
    Create and initialize agents based on the provided AgentSchema objects.

    Args:
        agents (List[AgentSchema]): A list of AgentSchema objects containing agent information.

    Returns:
        List[Agent]: The initialized Agent objects.

    """
    agent_list = []
    for agent in agents:
        name = agent.name
        system_prompt = agent.system_prompt

        logger.info(
            f"Creating agent: {name} with system prompt:"
            f" {system_prompt}"
        )

        out = Agent(
            agent_name=name,
            system_prompt=system_prompt,
            max_loops=1,
            autosave=True,
            dashboard=False,
            verbose=True,
            stopping_token="<DONE>",
            *args,
            **kwargs,
        )

        # Set the long term memory system of every agent to long term memory system
        agent_list.append(out)

    return agent_list
