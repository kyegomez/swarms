from typing import Callable
from swarms.schemas.agent_class_schema import AgentConfiguration
from swarms.tools.create_agent_tool import create_agent_tool
from swarms.prompts.agent_self_builder_prompt import (
    generate_agent_system_prompt,
)
from swarms.tools.base_tool import BaseTool
from swarms.structs.agent import Agent
import json


def self_agent_builder(
    task: str,
) -> Callable:
    schema = BaseTool().base_model_to_dict(AgentConfiguration)
    schema = [schema]

    print(json.dumps(schema, indent=4))

    prompt = generate_agent_system_prompt(task)

    agent = Agent(
        agent_name="Agent-Builder",
        agent_description="Autonomous agent builder",
        system_prompt=prompt,
        tools_list_dictionary=schema,
        output_type="final",
        max_loops=1,
        model_name="gpt-4o-mini",
    )

    agent_configuration = agent.run(
        f"Create the agent configuration for the task: {task}"
    )
    print(agent_configuration)
    print(type(agent_configuration))

    build_new_agent = create_agent_tool(agent_configuration)

    return build_new_agent
