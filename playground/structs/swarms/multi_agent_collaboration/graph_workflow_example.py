import os

from dotenv import load_dotenv

from swarms import Agent, Edge, GraphWorkflow, Node, NodeType, OpenAIChat

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

llm = OpenAIChat(temperature=0.5, openai_api_key=api_key, max_tokens=4000)
agent1 = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)
agent2 = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)


def sample_task():
    print("Running sample task")
    return "Task completed"


wf_graph = GraphWorkflow()
wf_graph.add_node(Node(id="agent1", type=NodeType.AGENT, agent=agent1))
wf_graph.add_node(Node(id="agent2", type=NodeType.AGENT, agent=agent2))
wf_graph.add_node(
    Node(id="task1", type=NodeType.TASK, callable=sample_task)
)
wf_graph.add_edge(Edge(source="agent1", target="task1"))
wf_graph.add_edge(Edge(source="agent2", target="task1"))

wf_graph.set_entry_points(["agent1", "agent2"])
wf_graph.set_end_points(["task1"])

print(wf_graph.visualize())

# Run the workflow
results = wf_graph.run()
print("Execution results:", results)
