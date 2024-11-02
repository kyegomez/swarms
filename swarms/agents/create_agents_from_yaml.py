import os
from typing import Any, Callable, Dict, List, Tuple, Union

import yaml
from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.swarm_router import SwarmRouter


def create_agents_from_yaml(
    model: Callable = None,
    yaml_file: str = "agents.yaml",
    return_type: str = "auto",
    *args,
    **kwargs,
) -> Union[
    SwarmRouter,
    Agent,
    List[Agent],
    Tuple[Union[SwarmRouter, Agent], List[Agent]],
    List[Dict[str, Any]],
]:
    """
    Create agents and/or SwarmRouter based on configurations defined in a YAML file.

    This function dynamically creates agents and a SwarmRouter (if specified) based on the
    configuration in the YAML file. It adapts its behavior based on the presence of a
    swarm architecture and the number of agents defined.

    Args:
        model (Callable): The language model to be used by the agents.
        yaml_file (str): Path to the YAML file containing agent and swarm configurations.
        return_type (str): Determines the return value. Options are:
                           "auto" (default): Automatically determine the most appropriate return type.
                           "swarm": Return SwarmRouter if present, otherwise a single agent or list of agents.
                           "agents": Return a list of agents (or a single agent if only one is defined).
                           "both": Return both SwarmRouter (or single agent) and list of agents.
                           "tasks": Return task results if any tasks were executed.
                           "run_swarm": Run the swarm and return its output.
        *args: Additional positional arguments for agent or SwarmRouter customization.
        **kwargs: Additional keyword arguments for agent or SwarmRouter customization.

    Returns:
        Union[SwarmRouter, Agent, List[Agent], Tuple[Union[SwarmRouter, Agent], List[Agent]], List[Dict[str, Any]]]:
        The return type depends on the 'return_type' argument and the configuration in the YAML file.

    Raises:
        FileNotFoundError: If the specified YAML file is not found.
        ValueError: If the YAML configuration is invalid or if an invalid return_type is specified.
    """
    try:
        logger.info(
            f"Checking if the YAML file {yaml_file} exists..."
        )

        if not os.path.exists(yaml_file):
            logger.error(f"YAML file {yaml_file} not found.")
            raise FileNotFoundError(
                f"YAML file {yaml_file} not found."
            )

        logger.info(f"Loading YAML file {yaml_file}")
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)

        if "agents" not in config:
            logger.error(
                "The YAML configuration does not contain 'agents'."
            )
            raise ValueError(
                "The YAML configuration does not contain 'agents'."
            )

        agents = []
        task_results = []

        # Create agents
        for agent_config in config["agents"]:
            logger.info(
                f"Creating agent: {agent_config['agent_name']}"
            )

            if "system_prompt" not in agent_config:
                logger.error(
                    f"System prompt is missing for agent: {agent_config['agent_name']}"
                )
                raise ValueError(
                    f"System prompt is missing for agent: {agent_config['agent_name']}"
                )

            agent = Agent(
                agent_name=agent_config["agent_name"],
                system_prompt=agent_config["system_prompt"],
                llm=model,
                max_loops=agent_config.get("max_loops", 1),
                autosave=agent_config.get("autosave", True),
                dashboard=agent_config.get("dashboard", False),
                verbose=agent_config.get("verbose", False),
                dynamic_temperature_enabled=agent_config.get(
                    "dynamic_temperature_enabled", False
                ),
                saved_state_path=agent_config.get("saved_state_path"),
                user_name=agent_config.get(
                    "user_name", "default_user"
                ),
                retry_attempts=agent_config.get("retry_attempts", 1),
                context_length=agent_config.get(
                    "context_length", 100000
                ),
                return_step_meta=agent_config.get(
                    "return_step_meta", False
                ),
                output_type=agent_config.get("output_type", "str"),
                auto_generate_prompt=agent_config.get(
                    "auto_generate_prompt", "False"
                ),
                *args,
                **kwargs,
            )

            logger.info(
                f"Agent {agent_config['agent_name']} created successfully."
            )
            agents.append(agent)

        # Create SwarmRouter if swarm_architecture is present
        swarm_router = None
        if "swarm_architecture" in config:
            swarm_config = config["swarm_architecture"]
            swarm_router = SwarmRouter(
                name=swarm_config["name"],
                description=swarm_config["description"],
                max_loops=swarm_config["max_loops"],
                agents=agents,
                swarm_type=swarm_config["swarm_type"],
                task=swarm_config.get("task"),
                flow=swarm_config.get("flow"),
                autosave=swarm_config.get("autosave"),
                return_json=swarm_config.get("return_json"),
                *args,
                **kwargs,
            )
            logger.info(
                f"SwarmRouter '{swarm_config['name']}' created successfully."
            )

        # Define function to run SwarmRouter
        def run_swarm_router(
            task: str = (
                swarm_config.get("task")
                if "swarm_architecture" in config
                else None
            ),
        ):
            if swarm_router:
                try:
                    output = swarm_router.run(task)
                    print(output)
                    logger.info(
                        f"Output for SwarmRouter '{swarm_config['name']}': {output}"
                    )
                    return output
                except Exception as e:
                    logger.error(
                        f"Error running task for SwarmRouter '{swarm_config['name']}': {e}"
                    )
                    raise e
            else:
                logger.error("SwarmRouter not created.")
                raise ValueError("SwarmRouter not created.")

        # Handle return types
        if return_type == "auto":
            if swarm_router:
                return swarm_router
            elif len(agents) == 1:
                return agents[0]
            else:
                return agents
        elif return_type == "swarm":
            return (
                swarm_router
                if swarm_router
                else (agents[0] if len(agents) == 1 else agents)
            )
        elif return_type == "agents":
            return agents[0] if len(agents) == 1 else agents
        elif return_type == "both":
            return (
                swarm_router
                if swarm_router
                else agents[0] if len(agents) == 1 else agents
            ), agents
        elif return_type == "tasks":
            if not task_results:
                logger.warning(
                    "No tasks were executed. Returning empty list."
                )
            return task_results
        elif return_type == "run_swarm":
            if swarm_router:
                return run_swarm_router()
            else:
                logger.error(
                    "Cannot run swarm: SwarmRouter not created."
                )
                raise ValueError(
                    "Cannot run swarm: SwarmRouter not created."
                )
        else:
            logger.error(f"Invalid return_type: {return_type}")
            raise ValueError(f"Invalid return_type: {return_type}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
