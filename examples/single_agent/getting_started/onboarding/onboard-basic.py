from dotenv import load_dotenv
from loguru import logger

from swarms import create_agents_from_yaml

# Load environment variables
load_dotenv()

# Path to your YAML file. Each agent's model is chosen by the model_name
# field inside the YAML, so no model object is built here.
yaml_file = "agents.yaml"

try:
    # Create agents and return them without running any task
    task_results = create_agents_from_yaml(
        yaml_file=yaml_file, return_type="agents"
    )

    print(task_results)
    logger.info(f"Results from agents: {task_results}")
except Exception as e:
    logger.error(f"An error occurred: {e}")
    print(e)
