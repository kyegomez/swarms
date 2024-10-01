import os

import yaml
from dotenv import load_dotenv
from loguru import logger
from swarm_models import OpenAIChat

from swarms.structs.agent import Agent

load_dotenv()


# Function to create and optionally run agents from a YAML file
def create_agents_from_yaml(
    yaml_file: str, return_type: str = "agents", *args, **kwargs
):
    """
    Create agents based on configurations defined in a YAML file.
    If a 'task' is provided in the YAML, the agent will execute the task after creation.

    Args:
        yaml_file (str): Path to the YAML file containing agent configurations.
        return_type (str): Determines the return value. "agents" to return agent list,
                           "tasks" to return task results, "both" to return both agents and tasks.
        *args: Additional positional arguments for agent customization.
        **kwargs: Additional keyword arguments for agent customization.

    Returns:
        List[Agent] or List[Task Results] or Tuple(List[Agent], List[Task Results])
    """
    logger.info(f"Checking if the YAML file {yaml_file} exists...")

    # Check if the YAML file exists
    if not os.path.exists(yaml_file):
        logger.error(f"YAML file {yaml_file} not found.")
        raise FileNotFoundError(f"YAML file {yaml_file} not found.")

    # Load the YAML configuration
    logger.info(f"Loading YAML file {yaml_file}")
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    # Ensure agents key exists
    if "agents" not in config:
        logger.error(
            "The YAML configuration does not contain 'agents'."
        )
        raise ValueError(
            "The YAML configuration does not contain 'agents'."
        )

    # List to store created agents and task results
    agents = []
    task_results = []

    # Iterate over each agent configuration and create agents
    for agent_config in config["agents"]:
        logger.info(f"Creating agent: {agent_config['agent_name']}")

        # Get the OpenAI API key from environment or YAML config
        api_key = os.getenv("OPENAI_API_KEY") or agent_config[
            "model"
        ].get("openai_api_key")
        if not api_key:
            logger.error(
                f"API key is missing for agent: {agent_config['agent_name']}"
            )
            raise ValueError(
                f"API key is missing for agent: {agent_config['agent_name']}"
            )

        # Create an instance of OpenAIChat model
        model = OpenAIChat(
            openai_api_key=api_key,
            model_name=agent_config["model"]["model_name"],
            temperature=agent_config["model"]["temperature"],
            max_tokens=agent_config["model"]["max_tokens"],
            *args,
            **kwargs,  # Pass any additional arguments to the model
        )

        # Ensure the system prompt is provided
        if "system_prompt" not in agent_config:
            logger.error(
                f"System prompt is missing for agent: {agent_config['agent_name']}"
            )
            raise ValueError(
                f"System prompt is missing for agent: {agent_config['agent_name']}"
            )

        # Dynamically choose the system prompt based on the agent config
        try:
            system_prompt = globals().get(
                agent_config["system_prompt"]
            )
            if not system_prompt:
                logger.error(
                    f"System prompt {agent_config['system_prompt']} not found."
                )
                raise ValueError(
                    f"System prompt {agent_config['system_prompt']} not found."
                )
        except KeyError:
            logger.error(
                f"System prompt {agent_config['system_prompt']} is not valid."
            )
            raise ValueError(
                f"System prompt {agent_config['system_prompt']} is not valid."
            )

        # Initialize the agent using the configuration
        agent = Agent(
            agent_name=agent_config["agent_name"],
            system_prompt=system_prompt,
            llm=model,
            max_loops=agent_config.get("max_loops", 1),
            autosave=agent_config.get("autosave", True),
            dashboard=agent_config.get("dashboard", False),
            verbose=agent_config.get("verbose", False),
            dynamic_temperature_enabled=agent_config.get(
                "dynamic_temperature_enabled", False
            ),
            saved_state_path=agent_config.get("saved_state_path"),
            user_name=agent_config.get("user_name", "default_user"),
            retry_attempts=agent_config.get("retry_attempts", 1),
            context_length=agent_config.get("context_length", 100000),
            return_step_meta=agent_config.get(
                "return_step_meta", False
            ),
            output_type=agent_config.get("output_type", "str"),
            *args,
            **kwargs,  # Pass any additional arguments to the agent
        )

        logger.info(
            f"Agent {agent_config['agent_name']} created successfully."
        )
        agents.append(agent)

        # Check if a task is provided, and if so, run the agent
        task = agent_config.get("task")
        if task:
            logger.info(
                f"Running task '{task}' with agent {agent_config['agent_name']}"
            )
            try:
                output = agent.run(task)
                logger.info(
                    f"Output for agent {agent_config['agent_name']}: {output}"
                )
                task_results.append(
                    {
                        "agent_name": agent_config["agent_name"],
                        "task": task,
                        "output": output,
                    }
                )
            except Exception as e:
                logger.error(
                    f"Error running task for agent {agent_config['agent_name']}: {e}"
                )
                task_results.append(
                    {
                        "agent_name": agent_config["agent_name"],
                        "task": task,
                        "error": str(e),
                    }
                )

    # Return results based on the `return_type`
    if return_type == "agents":
        return agents
    elif return_type == "tasks":
        return task_results
    elif return_type == "both":
        return agents, task_results
    else:
        logger.error(f"Invalid return_type: {return_type}")
        raise ValueError(f"Invalid return_type: {return_type}")


# # Usage example
# yaml_file = 'agents_config.yaml'

# try:
#     # Auto-create agents from the YAML file and return both agents and task results
#     agents, task_results = create_agents_from_yaml(yaml_file, return_type="tasks")

#     # Example: Print task results
#     for result in task_results:
#         print(f"Agent: {result['agent_name']} | Task: {result['task']} | Output: {result.get('output', 'Error encountered')}")

# except FileNotFoundError as e:
#     logger.error(f"Error: {e}")
# except ValueError as e:
#     logger.error(f"Error: {e}")
