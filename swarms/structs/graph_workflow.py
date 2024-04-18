import logging

from swarms.structs.base_structure import BaseStructure


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

    Examples:
    >>> from swarms.structs import GraphWorkflow
    >>> graph = GraphWorkflow()
    >>> graph.add("start", "Start")
    >>> graph.add("end", "End")
    >>> graph.start("start")
    """

    def __init__(self):
        self.graph = {}
        self.entry_point = None

    def add(self, node, node_value):
        """
        Adds a node to the graph with the specified value.

        Args:
            node (str): The name of the node.
            node_value (str): The value of the node.

        Returns:
            None
        """
        self.graph[node] = {"value": node_value, "edges": {}}
        logging.info(f"Added node: {node}")

    def start(self, node_name):
        """
        Sets the starting node for the workflow.

        Args:
            node_name (str): The name of the starting node.

        Returns:
            None
        """
        self._check_node_exists(node_name)

    def connect(self, from_node, to_node):
        """
        Connects two nodes in the graph.

        Args:
            from_node (str): The name of the source node.
            to_node (str): The name of the target node.

        Returns:
            None
        """
        self._check_node_exists(from_node, to_node)

    def set_entry_point(self, node_name):
        """
        Sets the entry point node for the workflow.

        Args:
            node_name (str): The name of the entry point node.

        Returns:
            None

        Raises:
            ValueError: If the specified node does not exist in the graph.
        """
        if node_name is self.graph:
            self.entry_point = node_name
        else:
            raise ValueError("Node does not exist in graph")

    def add_edge(self, from_node, to_node):
        """
        Adds an edge between two nodes in the graph.

        Args:
            from_node (str): The name of the source node.
            to_node (str): The name of the target node.

        Returns:
            None

        Raises:
            ValueError: If either the source or target node does not exist in the graph.
        """
        if from_node in self.graph and to_node in self.graph:
            self.graph[from_node]["edges"][to_node] = "edge"
        else:
            raise ValueError("Node does not exist in graph")

    def add_conditional_edges(self, from_node, condition, edge_dict):
        """
        Adds conditional edges from a node to multiple nodes based on a condition.

        Args:
            from_node (str): The name of the source node.
            condition: The condition for the conditional edges.
            edge_dict (dict): A dictionary mapping condition values to target nodes.

        Returns:
            None

        Raises:
            ValueError: If the source node or any of the target nodes do not exist in the graph.
        """
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
        """
        Runs the workflow and returns the graph.

        Returns:
            dict: The graph representing the nodes and edges.

        Raises:
            ValueError: If the entry point is not set.
        """
        if self.entry_point is None:
            raise ValueError("Entry point not set")
        return self.graph

    def _check_node_exists(self, node_name):
        """Checks if a node exists in the graph.

        Args:
            node_name (_type_): _description_

        Raises:
            ValueError: _description_
        """
        if node_name not in self.graph:
            raise ValueError(
                f"Node {node_name} does not exist in graph"
            )

    def _check_nodes_exist(self, from_node, to_node):
        """
        Checks if the given from_node and to_node exist in the graph.

        Args:
            from_node: The starting node of the edge.
            to_node: The ending node of the edge.

        Raises:
            NodeNotFoundError: If either from_node or to_node does not exist in the graph.
        """
        self._check_node_exists(from_node)
        self._check_node_exists(to_node)
