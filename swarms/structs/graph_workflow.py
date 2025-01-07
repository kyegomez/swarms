from enum import Enum
from typing import Any, Callable, Dict, List

import networkx as nx
from pydantic.v1 import BaseModel, Field, validator

from swarms.structs.agent import Agent  # noqa: F401
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="graph_workflow")


class NodeType(str, Enum):
    AGENT: Agent = "agent"
    TASK: str = "task"


class Node(BaseModel):
    """
    Represents a node in a graph workflow.

    Attributes:
        id (str): The unique identifier of the node.
        type (NodeType): The type of the node.
        callable (Callable, optional): The callable associated with the node. Required for task nodes.
        agent (Any, optional): The agent associated with the node.

    Raises:
        ValueError: If the node type is TASK and no callable is provided.

    Examples:
        >>> node = Node(id="task1", type=NodeType.TASK, callable=sample_task)
        >>> node = Node(id="agent1", type=NodeType.AGENT, agent=agent1)
        >>> node = Node(id="agent2", type=NodeType.AGENT, agent=agent2)

    """

    id: str
    type: NodeType
    callable: Callable = None
    agent: Any = None

    @validator("callable", always=True)
    def validate_callable(cls, value, values):
        if values["type"] == NodeType.TASK and value is None:
            raise ValueError("Task nodes must have a callable.")
        return value


class Edge(BaseModel):
    source: str
    target: str


class GraphWorkflow(BaseModel):
    """
    Represents a workflow graph.

    Attributes:
        nodes (Dict[str, Node]): A dictionary of nodes in the graph, where the key is the node ID and the value is the Node object.
        edges (List[Edge]): A list of edges in the graph, where each edge is represented by an Edge object.
        entry_points (List[str]): A list of node IDs that serve as entry points to the graph.
        end_points (List[str]): A list of node IDs that serve as end points of the graph.
        graph (nx.DiGraph): A directed graph object from the NetworkX library representing the workflow graph.
    """

    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: List[Edge] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    end_points: List[str] = Field(default_factory=list)
    graph: nx.DiGraph = Field(
        default_factory=nx.DiGraph, exclude=True
    )
    max_loops: int = 1

    class Config:
        arbitrary_types_allowed = True

    def add_node(self, node: Node):
        """
        Adds a node to the workflow graph.

        Args:
            node (Node): The node object to be added.

        Raises:
            ValueError: If a node with the same ID already exists in the graph.
        """
        try:
            if node.id in self.nodes:
                raise ValueError(
                    f"Node with id {node.id} already exists."
                )
            self.nodes[node.id] = node
            self.graph.add_node(
                node.id,
                type=node.type,
                callable=node.callable,
                agent=node.agent,
            )
        except Exception as e:
            logger.info(f"Error in adding node to the workflow: {e}")
            raise e

    def add_edge(self, edge: Edge):
        """
        Adds an edge to the workflow graph.

        Args:
            edge (Edge): The edge object to be added.

        Raises:
            ValueError: If either the source or target node of the edge does not exist in the graph.
        """
        if (
            edge.source not in self.nodes
            or edge.target not in self.nodes
        ):
            raise ValueError(
                "Both source and target nodes must exist before adding an edge."
            )
        self.edges.append(edge)
        self.graph.add_edge(edge.source, edge.target)

    def set_entry_points(self, entry_points: List[str]):
        """
        Sets the entry points of the workflow graph.

        Args:
            entry_points (List[str]): A list of node IDs to be set as entry points.

        Raises:
            ValueError: If any of the specified node IDs do not exist in the graph.
        """
        for node_id in entry_points:
            if node_id not in self.nodes:
                raise ValueError(
                    f"Node with id {node_id} does not exist."
                )
        self.entry_points = entry_points

    def set_end_points(self, end_points: List[str]):
        """
        Sets the end points of the workflow graph.

        Args:
            end_points (List[str]): A list of node IDs to be set as end points.

        Raises:
            ValueError: If any of the specified node IDs do not exist in the graph.
        """
        for node_id in end_points:
            if node_id not in self.nodes:
                raise ValueError(
                    f"Node with id {node_id} does not exist."
                )
        self.end_points = end_points

    def visualize(self) -> str:
        """
        Generates a string representation of the workflow graph in the Mermaid syntax.

        Returns:
            str: The Mermaid string representation of the workflow graph.
        """
        mermaid_str = "graph TD\n"
        for node_id, node in self.nodes.items():
            mermaid_str += f"    {node_id}[{node_id}]\n"
        for edge in self.edges:
            mermaid_str += f"    {edge.source} --> {edge.target}\n"
        return mermaid_str

    def run(
        self, task: str = None, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Function to run the workflow graph.

        Args:
            task (str): The task to be executed by the workflow.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the execution.

        Raises:
            ValueError: If no entry points or end points are defined in the graph.

        """
        try:
            loop = 0
            while loop < self.max_loops:
                # Ensure all nodes and edges are valid
                if not self.entry_points:
                    raise ValueError(
                        "At least one entry point must be defined."
                    )
                if not self.end_points:
                    raise ValueError(
                        "At least one end point must be defined."
                    )

                # Perform a topological sort of the graph to ensure proper execution order
                sorted_nodes = list(nx.topological_sort(self.graph))

                # Initialize execution state
                execution_results = {}

                for node_id in sorted_nodes:
                    node = self.nodes[node_id]
                    if node.type == NodeType.TASK:
                        print(f"Executing task: {node_id}")
                        result = node.callable()
                    elif node.type == NodeType.AGENT:
                        print(f"Executing agent: {node_id}")
                        result = node.agent.run(task, *args, **kwargs)
                    execution_results[node_id] = result

                loop += 1

                return execution_results
        except Exception as e:
            logger.info(f"Error in running the workflow: {e}")
            raise e


# # Example usage
# if __name__ == "__main__":
#     from swarms import Agent

#     import os
#     from dotenv import load_dotenv

#     load_dotenv()

#     api_key = os.environ.get("OPENAI_API_KEY")

#     llm = OpenAIChat(
#         temperature=0.5, openai_api_key=api_key, max_tokens=4000
#     )
#     agent1 = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)
#     agent2 = Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)

#     def sample_task():
#         print("Running sample task")
#         return "Task completed"

#     wf_graph = GraphWorkflow()
#     wf_graph.add_node(Node(id="agent1", type=NodeType.AGENT, agent=agent1))
#     wf_graph.add_node(Node(id="agent2", type=NodeType.AGENT, agent=agent2))
#     wf_graph.add_node(
#         Node(id="task1", type=NodeType.TASK, callable=sample_task)
#     )
#     wf_graph.add_edge(Edge(source="agent1", target="task1"))
#     wf_graph.add_edge(Edge(source="agent2", target="task1"))

#     wf_graph.set_entry_points(["agent1", "agent2"])
#     wf_graph.set_end_points(["task1"])

#     print(wf_graph.visualize())

#     # Run the workflow
#     results = wf_graph.run()
#     print("Execution results:", results)
