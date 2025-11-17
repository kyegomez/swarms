#!/usr/bin/env python3
"""
Basic Graph Workflow Example

A minimal example showing how to use GraphWorkflow with backend selection.
"""

from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

agent_one = Agent(agent_name="research_agent", model="gpt-4o-mini")
agent_two = Agent(
    agent_name="research_agent_two", model="gpt-4o-mini"
)
agent_three = Agent(
    agent_name="research_agent_three", model="gpt-4o-mini"
)


def main():
    """
    Run a basic graph workflow example without print statements.
    """
    # Create agents

    # Create workflow with backend selection
    workflow = GraphWorkflow(
        name="Basic Example",
        verbose=True,
    )

    # Add agents to workflow
    workflow.add_node(agent_one)
    workflow.add_node(agent_two)
    workflow.add_node(agent_three)

    # Create simple chain using the actual agent names
    workflow.add_edge("research_agent", "research_agent_two")
    workflow.add_edge("research_agent_two", "research_agent_three")

    # Compile the workflow
    workflow.compile()

    # Run the workflow
    task = "Complete a simple task"
    results = workflow.run(task)
    return results


if __name__ == "__main__":
    main()
