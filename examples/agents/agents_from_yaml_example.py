from loguru import logger
from dotenv import load_dotenv
from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)

# Load environment variables
load_dotenv()

# Path to your YAML file
yaml_file = "agents.yaml"

try:
    # Create agents and run tasks (using 'both' to return agents and task results)
    task_results = create_agents_from_yaml(
        yaml_file, return_type="tasks"
    )

    logger.info(f"Results from agents: {task_results}")
except Exception as e:
    logger.error(f"An error occurred: {e}")
