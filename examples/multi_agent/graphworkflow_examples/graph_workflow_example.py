from swarms import Agent, GraphWorkflow
from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT_TWO,
)

# Define two real agents with the multi-agent collaboration prompt
agent1 = Agent(
    agent_name="ResearchAgent1",
    model_name="gpt-4.1",
    max_loops=1,
    system_prompt=MULTI_AGENT_COLLAB_PROMPT_TWO,  # Set collaboration prompt
)

agent2 = Agent(
    agent_name="ResearchAgent2",
    model_name="gpt-4.1",
    max_loops=1,
    system_prompt=MULTI_AGENT_COLLAB_PROMPT_TWO,  # Set collaboration prompt
)

# Build the workflow with only agents as nodes
workflow = GraphWorkflow(
    name="Research Workflow",
    description="A workflow for researching the best arbitrage trading strategies for altcoins",
    auto_compile=True,
)
workflow.add_node(agent1)
workflow.add_node(agent2)

# Define a relationship: agent1 feeds into agent2
workflow.add_edge(agent1.agent_name, agent2.agent_name)

# Visualize the workflow using Graphviz
workflow.visualize()

workflow.compile()

# Export workflow to JSON
workflow_json = workflow.to_json()
print(workflow_json)

# Run the workflow and print results
results = workflow.run(
    task="What are the best arbitrage trading strategies for altcoins? Give me research papers and articles on the topic."
)
print("\n📋 Execution results:")
for agent_name, result in results.items():
    print(f"\n🤖 {agent_name}:")
    print(f"  {result[:200]}{'...' if len(result) > 200 else ''}")
