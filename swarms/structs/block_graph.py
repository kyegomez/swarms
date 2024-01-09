from swarms.structs.base import BaseStructure


class GraphWorkflow(BaseStructure):
    """
    Represents a graph-based workflow structure.

    Attributes:
        graph (dict): A dictionary representing the nodes and edges of the graph.
        entry_point (str): The name of the entry point node in the graph.

    Methods:
        add(node, node_value): Adds a node to the graph with the specified value.
        start(node_name): Sets the starting node for the workflow.
        connect(from_node, to_node): Connects two nodes in the graph.
        set_entry_point(node_name): Sets the entry point node for the workflow.
        add_edge(from_node, to_node): Adds an edge between two nodes in the graph.
        add_conditional_edges(from_node, condition, edge_dict): Adds conditional edges from a node to multiple nodes based on a condition.
        run(): Runs the workflow and returns the graph.
    """

    def __init__(self):
        self.graph = {}
        self.entry_point = None

    def add(self, node, node_value):
        self.graph[node] = {"value": node_value, "edges": {}}

    def start(self, node_name):
        self._check_node_exists(node_name)

    def connect(self, from_node, to_node):
        self._check_node_exists(from_node, to_node)

    def set_entry_point(self, node_name):
        if node_name is self.graph:
            self.entry_point = node_name
        else:
            raise ValueError("Node does not exist in graph")

    def add_edge(self, from_node, to_node):
        if from_node in self.graph and to_node in self.graph:
            self.graph[from_node]["edges"][to_node] = "edge"
        else:
            raise ValueError("Node does not exist in graph")

    def add_conditional_edges(self, from_node, condition, edge_dict):
        if from_node in self.graph:
            for condition_value, to_node in edge_dict.items():
                if to_node in self.graph:
                    self.graph[from_node]["edges"][
                        to_node
                    ] = condition
                else:
                    raise ValueError("Node does not exist in graph")
        else:
            raise ValueError(
                f"Node {from_node} does not exist in graph"
            )

    def run(self):
        if self.entry_point is None:
            raise ValueError("Entry point not set")
        return self.graph

    def _check_node_exists(self, node_name):
        if node_name not in self.graph:
            raise ValueError(
                f"Node {node_name} does not exist in graph"
            )

    def _check_nodes_exist(self, from_node, to_node):
        self._check_node_exists(from_node)
        self._check_node_exists(to_node)
