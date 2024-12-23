import os
from datetime import datetime
from uuid import uuid4

# Import necessary classes from your swarm module
from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.telemetry.capture_sys_data import log_agent_data
from swarms.utils.file_processing import create_file_in_folder
from swarms import SpreadSheetSwarm

# Ensure you have an environment variable or default workspace dir
workspace_dir = os.getenv("WORKSPACE_DIR", "./workspace")

def create_agents(num_agents: int):
    """
    Create a list of agent instances.
    
    Args:
        num_agents (int): The number of agents to create.
    
    Returns:
        List[Agent]: List of created Agent objects.
    """
    agents = []
    for i in range(num_agents):
        agent_name = f"Agent-{i + 1}"
        agents.append(Agent(agent_name=agent_name))
    return agents

def main():
    # Number of agents to create
    num_agents = 5

    # Create the agents
    agents = create_agents(num_agents)

    # Initialize the swarm with agents and other configurations
    swarm = SpreadSheetSwarm(
        name="Test-Swarm",
        description="A swarm for testing purposes.",
        agents=agents,
        autosave_on=True,
        max_loops=2,
        workspace_dir=workspace_dir
    )

    # Run a sample task in the swarm (synchronously)
    task = "process_data"
    
    # Ensure the run method is synchronous
    swarm_metadata = swarm.run(task)  # Assuming this is made synchronous

    # Print swarm metadata after task completion
    print("Swarm Metadata:")
    print(swarm_metadata)

    # Check if CSV file has been created and saved
    if os.path.exists(swarm.save_file_path):
        print(f"Metadata saved to: {swarm.save_file_path}")
    else:
        print(f"Metadata not saved correctly. Check the save path.")

    # Test saving metadata to JSON file
    swarm.data_to_json_file()

    # Test exporting metadata to JSON
    swarm_json = swarm.export_to_json()
    print("Exported JSON metadata:")
    print(swarm_json)

    # Log agent data
    print("Logging agent data:")
    print(log_agent_data(swarm.metadata.model_dump()))

# Run the synchronous main function
if __name__ == "__main__":
    main()
