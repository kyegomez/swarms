from swarms import Agent, Edge, GraphWorkflow, Node, NodeType
from swarms.prompts.finance_agent_sys_prompt import (
    FINANCIAL_AGENT_SYS_PROMPT,
)

# Initialize agents with model_name parameter
agent1 = Agent(
    agent_name="Agent1",
    model_name="openai/gpt-4o-mini",  # Using provider prefix
    temperature=0.5,
    max_tokens=4000,
    max_loops=1,
    autosave=True,
    dashboard=True,
)

agent2 = Agent(
    agent_name="Agent2",
    model_name="openai/gpt-4o-mini",  # Using provider prefix
    temperature=0.5,
    max_tokens=4000,
    max_loops=1,
    autosave=True,
    dashboard=True,
)

def sample_task():
    print("Running sample task")
    return "Task completed"

wf_graph = GraphWorkflow()
wf_graph.add_node(Node(id="agent1", type=NodeType.AGENT, agent=agent1))
wf_graph.add_node(Node(id="agent2", type=NodeType.AGENT, agent=agent2))
wf_graph.add_node(Node(id="task1", type=NodeType.TASK, callable=sample_task))

wf_graph.add_edge(Edge(source="agent1", target="task1"))
wf_graph.add_edge(Edge(source="agent2", target="task1"))

wf_graph.set_entry_points(["agent1", "agent2"])
wf_graph.set_end_points(["task1"])

print(wf_graph.visualize())

results = wf_graph.run()
print("Execution results:", results)