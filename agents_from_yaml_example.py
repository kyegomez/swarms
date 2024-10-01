from loguru import logger
from dotenv import load_dotenv
from swarms import create_agents_from_yaml

# Load environment variables
load_dotenv()

# Path to your YAML file
yaml_file = "agents_config.yaml"

try:
    # Create agents and run tasks (using 'both' to return agents and task results)
    agents, task_results = create_agents_from_yaml(
        yaml_file, return_type="both"
    )

    # Print the results of the tasks
    for result in task_results:
        print(
            f"Agent: {result['agent_name']} | Task: {result['task']} | Output: {result.get('output', 'Error encountered')}"
        )

except Exception as e:
    logger.error(f"An error occurred: {e}")
