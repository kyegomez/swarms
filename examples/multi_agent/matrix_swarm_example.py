from swarms import Agent
from swarms.structs.matrix_swarm import MatrixSwarm
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT


def create_agent(name: str) -> Agent:
    """Utility function to build a simple agent for the matrix."""
    return Agent(
        agent_name=name,
        system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
        model_name="gpt-4o-mini",
        max_loops=1,
        streaming_on=False,
        verbose=True,
    )


if __name__ == "__main__":
    # Initialize a 2x2 matrix of agents
    agents = [
        [create_agent(f"Agent-{i}-{j}") for j in range(2)]
        for i in range(2)
    ]
    swarm = MatrixSwarm(agents)

    # Perform basic matrix operations
    transposed = swarm.transpose()
    added = swarm.add(transposed)

    # Show shapes after operations
    print("Original shape:", len(swarm.agents), len(swarm.agents[0]))
    print("Transposed shape:", len(transposed.agents), len(transposed.agents[0]))
    print("Added matrix shape:", len(added.agents), len(added.agents[0]))

    # Prepare queries for each row of the matrix
    queries = [
        "What are the benefits of index funds?",
        "How does compound interest work?",
    ]

    # Run agents by multiplying the matrix with its transpose
    results = swarm.multiply(transposed, queries)

    # Display results
    for row in results:
        for output in row:
            print(f"{output.agent_name}: {output.output_result}")

