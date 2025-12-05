from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

agent_one = Agent(
    agent_name="research_agent",
    model_name="gpt-4o-mini",
    name="Research Agent",
    agent_description="Agent responsible for gathering and summarizing research information.",
)
agent_two = Agent(
    agent_name="research_agent_two",
    model_name="gpt-4o-mini",
    name="Analysis Agent",
    agent_description="Agent that analyzes the research data provided and processes insights.",
)
agent_three = Agent(
    agent_name="research_agent_three",
    model_name="gpt-4o-mini",
    agent_description="Agent tasked with structuring analysis into a final report or output.",
)

# Create workflow with backend selection
workflow = GraphWorkflow(
    name="Basic Example",
    verbose=True,
)

workflow.add_nodes([agent_one, agent_two, agent_three])

# Create simple chain using the actual agent names
workflow.add_edge("research_agent", "research_agent_two")
workflow.add_edge("research_agent_two", "research_agent_three")

workflow.visualize()

# Compile the workflow
workflow.compile()

# Run the workflow
task = "Complete a simple task"
results = workflow.run(task)

print(results)
