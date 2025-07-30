from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow
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
workflow = GraphWorkflow()
workflow.add_node(agent1)
workflow.add_node(agent2)

# Define a relationship: agent1 feeds into agent2
workflow.add_edge(agent1.agent_name, agent2.agent_name)

# Visualize the workflow using Graphviz
print("\n📊 Creating workflow visualization...")
try:
    viz_output = workflow.visualize(
        output_path="simple_workflow_graph",
        format="png",
        view=True,  # Auto-open the generated image
        show_parallel_patterns=True,
    )
    print(f"✅ Workflow visualization saved to: {viz_output}")
except Exception as e:
    print(f"⚠️ Graphviz not available, using text visualization: {e}")
    workflow.visualize()

# Export workflow to JSON
workflow_json = workflow.to_json()
print(
    f"\n💾 Workflow exported to JSON ({len(workflow_json)} characters)"
)

# Run the workflow and print results
print("\n🚀 Executing workflow...")
results = workflow.run(
    task="What are the best arbitrage trading strategies for altcoins? Give me research papers and articles on the topic."
)
print("\n📋 Execution results:")
for agent_name, result in results.items():
    print(f"\n🤖 {agent_name}:")
    print(f"  {result[:200]}{'...' if len(result) > 200 else ''}")
