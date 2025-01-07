import os

from swarms import SpreadSheetSwarm

# Create the swarm
swarm = SpreadSheetSwarm(
    name="Financial-Analysis-Swarm",
    description="A swarm of agents performing concurrent financial analysis tasks",
    max_loops=1,
    workspace_dir="./workspace",
    load_path="swarm.csv",
)

try:
    # Ensure workspace directory exists
    os.makedirs("./workspace", exist_ok=True)

    # Load the financial analysts from CSV
    swarm.load_from_csv()

    print(f"Loaded {len(swarm.agents)} financial analysis agents")
    print("\nStarting concurrent financial analysis tasks...")

    # Run all agents concurrently with their configured tasks
    results = swarm.run()

    print(
        "\nAnalysis complete! Results saved to:", swarm.save_file_path
    )
    print("\nSwarm execution metadata:")
    print(results)

except Exception as e:
    print(f"An error occurred: {e!s}")
