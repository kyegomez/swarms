from swarms import Agent, ConcurrentWorkflow
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

if __name__ == "__main__":
    # Assuming you've already initialized some agents outside of this class
    agents = [
        Agent(
            agent_name=f"Financial-Analysis-Agent-{i}",
            system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
            model_name="gpt-4.1",
            max_loops=1,
        )
        for i in range(3)  # Adjust number of agents as needed
    ]

    # Initialize the workflow with the list of agents
    workflow = ConcurrentWorkflow(
        agents=agents,
        output_type="list",
        max_loops=1,
    )

    # Define the task for all agents
    task = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"

    # Run the workflow and save metadata
    metadata = workflow.run(task)
    print(metadata)
