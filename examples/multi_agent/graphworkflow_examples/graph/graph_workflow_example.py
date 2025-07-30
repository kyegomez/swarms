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

# print(workflow.to_json())

print(workflow.visualize())

# Optionally, run the workflow and print the results
# results = workflow.run(
#     task="What are the best arbitrage trading strategies for altcoins? Give me research papers and articles on the topic."
# )
# print("Execution results:", results)
