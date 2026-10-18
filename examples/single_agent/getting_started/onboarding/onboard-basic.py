from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from swarms import create_agents_from_yaml

# Load environment variables
load_dotenv()

# Load the bundled YAML beside this script. Each agent uses its
# model_name field, so no model object is built here.
yaml_file = str(Path(__file__).resolve().with_name("agents.yaml"))

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
