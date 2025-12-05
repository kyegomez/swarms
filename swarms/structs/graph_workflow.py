import asyncio
import concurrent.futures
import json
import os
import time
import traceback
import uuid
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import networkx as nx

try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    graphviz = None

try:
    import rustworkx as rx

    RUSTWORKX_AVAILABLE = True
except ImportError:
    RUSTWORKX_AVAILABLE = False
    rx = None

from swarms.structs.agent import Agent  # noqa: F401
from swarms.structs.conversation import Conversation
from swarms.utils.get_cpu_cores import get_cpu_cores
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="graph_workflow")


class GraphBackend:
    """
    Abstract base class for graph backends.
    Provides a unified interface for different graph libraries.
    """

    def add_node(self, node_id: str, **attrs) -> None:
        """
        Add a node to the graph.

        Args:
            node_id (str): The unique identifier of the node.
            **attrs: Additional attributes for the node.
        """
        raise NotImplementedError

    def add_edge(self, source: str, target: str, **attrs) -> None:
        """
        Add an edge to the graph.

        Args:
            source (str): The source node ID.
            target (str): The target node ID.
            **attrs: Additional attributes for the edge.
        """
        raise NotImplementedError

    def in_degree(self, node_id: str) -> int:
        """
        Get the in-degree of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            int: The in-degree of the node.
        """
        raise NotImplementedError

    def out_degree(self, node_id: str) -> int:
        """
        Get the out-degree of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            int: The out-degree of the node.
        """
        raise NotImplementedError

    def predecessors(self, node_id: str) -> Iterator[str]:
        """
        Get the predecessors of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            Iterator[str]: Iterator of predecessor node IDs.
        """
        raise NotImplementedError

    def reverse(self) -> "GraphBackend":
        """
        Return a reversed copy of the graph.

        Returns:
            GraphBackend: A new backend instance with reversed edges.
        """
        raise NotImplementedError

    def topological_generations(self) -> List[List[str]]:
        """
        Get topological generations (layers) of the graph.

        Returns:
            List[List[str]]: List of layers, where each layer is a list of node IDs.
        """
        raise NotImplementedError

    def simple_cycles(self) -> List[List[str]]:
        """
        Find simple cycles in the graph.

        Returns:
            List[List[str]]: List of cycles, where each cycle is a list of node IDs.
        """
        raise NotImplementedError

    def descendants(self, node_id: str) -> Set[str]:
        """
        Get all descendants of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            Set[str]: Set of descendant node IDs.
        """
        raise NotImplementedError


class NetworkXBackend(GraphBackend):
    """
    NetworkX backend implementation.
    """

    def __init__(self):
        """
        Initialize the NetworkX backend.
        """
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, **attrs) -> None:
        """
        Add a node to the NetworkX graph.

        Args:
            node_id (str): The unique identifier of the node.
            **attrs: Additional attributes for the node.
        """
        self.graph.add_node(node_id, **attrs)

    def add_edge(self, source: str, target: str, **attrs) -> None:
        """
        Add an edge to the NetworkX graph.

        Args:
            source (str): The source node ID.
            target (str): The target node ID.
            **attrs: Additional attributes for the edge.
        """
        self.graph.add_edge(source, target, **attrs)

    def in_degree(self, node_id: str) -> int:
        """
        Get the in-degree of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            int: The in-degree of the node.
        """
        return self.graph.in_degree(node_id)

    def out_degree(self, node_id: str) -> int:
        """
        Get the out-degree of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            int: The out-degree of the node.
        """
        return self.graph.out_degree(node_id)

    def predecessors(self, node_id: str) -> Iterator[str]:
        """
        Get the predecessors of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            Iterator[str]: Iterator of predecessor node IDs.
        """
        return self.graph.predecessors(node_id)

    def reverse(self) -> "NetworkXBackend":
        """
        Return a reversed copy of the graph.

        Returns:
            NetworkXBackend: A new backend instance with reversed edges.
        """
        reversed_backend = NetworkXBackend()
        reversed_backend.graph = self.graph.reverse()
        return reversed_backend

    def topological_generations(self) -> List[List[str]]:
        """
        Get topological generations (layers) of the graph.

        Returns:
            List[List[str]]: List of layers, where each layer is a list of node IDs.
        """
        return list(nx.topological_generations(self.graph))

    def simple_cycles(self) -> List[List[str]]:
        """
        Find simple cycles in the graph.

        Returns:
            List[List[str]]: List of cycles, where each cycle is a list of node IDs.
        """
        return list(nx.simple_cycles(self.graph))

    def descendants(self, node_id: str) -> Set[str]:
        """
        Get all descendants of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            Set[str]: Set of descendant node IDs.
        """
        return nx.descendants(self.graph, node_id)


class RustworkxBackend(GraphBackend):
    """
    Rustworkx backend implementation.
    Uses integer indices internally but exposes string node IDs.
    """

    def __init__(self):
        """
        Initialize the Rustworkx backend.
        """
        if not RUSTWORKX_AVAILABLE:
            raise ImportError(
                "rustworkx is not installed. Install it with: pip install rustworkx"
            )
        self.graph = rx.PyDiGraph()
        # Mapping from node ID (string) to node index (int)
        self._node_id_to_index: Dict[str, int] = {}
        # Mapping from node index (int) to node ID (string)
        self._index_to_node_id: Dict[int, str] = {}

    def _get_or_create_node_index(self, node_id: str) -> int:
        """
        Get the node index for a given node ID, creating it if necessary.

        Args:
            node_id (str): The node ID.

        Returns:
            int: The node index.
        """
        if node_id not in self._node_id_to_index:
            node_index = self.graph.add_node(node_id)
            self._node_id_to_index[node_id] = node_index
            self._index_to_node_id[node_index] = node_id
        return self._node_id_to_index[node_id]

    def add_node(self, node_id: str, **attrs) -> None:
        """
        Add a node to the Rustworkx graph.

        Args:
            node_id (str): The unique identifier of the node.
            **attrs: Additional attributes for the node (stored in node data).
        """
        if node_id not in self._node_id_to_index:
            # Store node data as a dict with the node_id and attributes
            node_data = {"node_id": node_id, **attrs}
            node_index = self.graph.add_node(node_data)
            self._node_id_to_index[node_id] = node_index
            self._index_to_node_id[node_index] = node_id
        else:
            # Update existing node data
            node_index = self._node_id_to_index[node_id]
            node_data = self.graph[node_index]
            if isinstance(node_data, dict):
                node_data.update(attrs)
            else:
                self.graph[node_index] = {"node_id": node_id, **attrs}

    def add_edge(self, source: str, target: str, **attrs) -> None:
        """
        Add an edge to the Rustworkx graph.

        Args:
            source (str): The source node ID.
            target (str): The target node ID.
            **attrs: Additional attributes for the edge (stored in edge data).
        """
        source_idx = self._get_or_create_node_index(source)
        target_idx = self._get_or_create_node_index(target)
        edge_data = attrs if attrs else None
        self.graph.add_edge(source_idx, target_idx, edge_data)

    def in_degree(self, node_id: str) -> int:
        """
        Get the in-degree of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            int: The in-degree of the node.
        """
        if node_id not in self._node_id_to_index:
            return 0
        node_index = self._node_id_to_index[node_id]
        return self.graph.in_degree(node_index)

    def out_degree(self, node_id: str) -> int:
        """
        Get the out-degree of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            int: The out-degree of the node.
        """
        if node_id not in self._node_id_to_index:
            return 0
        node_index = self._node_id_to_index[node_id]
        return self.graph.out_degree(node_index)

    def predecessors(self, node_id: str) -> Iterator[str]:
        """
        Get the predecessors of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            Iterator[str]: Iterator of predecessor node IDs.
        """
        if node_id not in self._node_id_to_index:
            return iter([])
        target_index = self._node_id_to_index[node_id]
        # Use edge list to find predecessors (more reliable than predecessors() method)
        result = []
        for edge in self.graph.edge_list():
            source_idx, target_idx = edge
            if target_idx == target_index:
                result.append(self._index_to_node_id[source_idx])
        return iter(result)

    def reverse(self) -> "RustworkxBackend":
        """
        Return a reversed copy of the graph.

        Returns:
            RustworkxBackend: A new backend instance with reversed edges.
        """
        reversed_backend = RustworkxBackend()
        # Copy the graph structure
        reversed_backend.graph = self.graph.copy()
        # Reverse the edges
        reversed_backend.graph.reverse()
        # Copy the mappings
        reversed_backend._node_id_to_index = (
            self._node_id_to_index.copy()
        )
        reversed_backend._index_to_node_id = (
            self._index_to_node_id.copy()
        )
        return reversed_backend

    def topological_generations(self) -> List[List[str]]:
        """
        Get topological generations (layers) of the graph.

        Returns:
            List[List[str]]: List of layers, where each layer is a list of node IDs.
        """
        try:
            # Get all node indices
            all_indices = list(self._node_id_to_index.values())
            if not all_indices:
                return []

            # Use layer-by-layer approach similar to NetworkX topological_generations
            layers = []
            remaining = set(all_indices)
            processed = set()

            while remaining:
                # Find all nodes with in-degree 0 considering only edges from processed nodes
                # In rustworkx, we need to check if all predecessors are in processed set
                layer = []
                # First pass: identify nodes that can be added to this layer
                # (without modifying remaining/processed during iteration)
                nodes_to_add = []
                for idx in list(remaining):
                    # Get all predecessors using edge list
                    pred_indices = []
                    for edge in self.graph.edge_list():
                        source_idx, target_idx = edge
                        if target_idx == idx:
                            pred_indices.append(source_idx)
                    # Check if all predecessors have been processed (or node has no predecessors)
                    # A node can be added to the layer if:
                    # 1. It has no predecessors (entry node), OR
                    # 2. All its predecessors have already been processed (from previous layers)
                    if not pred_indices:
                        # No predecessors - this is an entry node
                        nodes_to_add.append(idx)
                    elif all(
                        pred_idx in processed
                        for pred_idx in pred_indices
                    ):
                        # All predecessors have been processed in previous layers
                        nodes_to_add.append(idx)

                # Second pass: add identified nodes to the layer and update sets
                for idx in nodes_to_add:
                    layer.append(self._index_to_node_id[idx])
                    remaining.remove(idx)
                    processed.add(idx)

                if not layer:
                    # Cycle detected or error, break
                    break

                layers.append(layer)

            # If there are remaining nodes, they form a cycle - add them as a final layer
            if remaining:
                cycle_layer = [
                    self._index_to_node_id[idx] for idx in remaining
                ]
                layers.append(cycle_layer)

            return (
                layers
                if layers
                else [
                    [
                        self._index_to_node_id[idx]
                        for idx in all_indices
                    ]
                ]
            )
        except Exception as e:
            logger.warning(
                f"Error in rustworkx topological_generations: {e}, falling back to simple approach"
            )
            # Fallback: return all nodes in one layer
            return [
                [node_id for node_id in self._node_id_to_index.keys()]
            ]

    def simple_cycles(self) -> List[List[str]]:
        """
        Find simple cycles in the graph.

        Returns:
            List[List[str]]: List of cycles, where each cycle is a list of node IDs.
        """
        try:
            # Convert to NetworkX temporarily for cycle detection
            # This is a limitation of rustworkx - it doesn't have simple_cycles
            # We'll use a workaround by converting temporarily
            import networkx as nx

            nx_graph = nx.DiGraph()
            for node_id in self._node_id_to_index.keys():
                nx_graph.add_node(node_id)
            for edge in self.graph.edge_list():
                source_idx, target_idx = edge
                source_id = self._index_to_node_id[source_idx]
                target_id = self._index_to_node_id[target_idx]
                nx_graph.add_edge(source_id, target_id)

            cycles = list(nx.simple_cycles(nx_graph))
            return cycles
        except Exception as e:
            logger.warning(
                f"Error in rustworkx simple_cycles: {e}, returning empty list"
            )
            return []

    def descendants(self, node_id: str) -> Set[str]:
        """
        Get all descendants of a node.

        Args:
            node_id (str): The node ID.

        Returns:
            Set[str]: Set of descendant node IDs.
        """
        if node_id not in self._node_id_to_index:
            return set()
        node_index = self._node_id_to_index[node_id]
        # Use BFS to find all descendants
        descendants = set()
        queue = [node_index]
        visited = {node_index}

        while queue:
            current_idx = queue.pop(0)
            succ_data = self.graph.successors(current_idx)
            for succ in succ_data:
                # Handle both dict (node data) and int (index) returns
                if isinstance(succ, dict):
                    succ_node_id = succ.get("node_id")
                    if (
                        succ_node_id
                        and succ_node_id in self._node_id_to_index
                    ):
                        succ_idx = self._node_id_to_index[
                            succ_node_id
                        ]
                    else:
                        continue
                elif isinstance(succ, int):
                    succ_idx = succ
                else:
                    continue

                if succ_idx not in visited:
                    visited.add(succ_idx)
                    descendants.add(self._index_to_node_id[succ_idx])
                    queue.append(succ_idx)

        return descendants


class NodeType(str, Enum):
    AGENT: Agent = "agent"


class Node:
    """
    Represents a node in a graph workflow. Only agent nodes are supported.

    Attributes:
        id (str): The unique identifier of the node.
        type (NodeType): The type of the node (always AGENT).
        agent (Any): The agent associated with the node.
        metadata (Dict[str, Any], optional): Additional metadata for the node.
    """

    def __init__(
        self,
        id: str = None,
        type: NodeType = NodeType.AGENT,
        agent: Any = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize a Node.

        Args:
            id (str, optional): The unique identifier of the node.
            type (NodeType, optional): The type of the node. Defaults to NodeType.AGENT.
            agent (Any, optional): The agent associated with the node.
            metadata (Dict[str, Any], optional): Additional metadata for the node.
        """
        self.id = id
        self.type = type
        self.agent = agent
        self.metadata = metadata or {}

        if not self.id:
            if self.type == NodeType.AGENT and self.agent is not None:
                self.id = getattr(self.agent, "agent_name", None)
            if not self.id:
                raise ValueError(
                    "Node id could not be auto-assigned. Please provide an id."
                )

    @classmethod
    def from_agent(cls, agent: Agent, **kwargs: Any) -> "Node":
        """
        Create a Node from an Agent object.

        Args:
            agent (Agent): The agent to create a node from.
            **kwargs: Additional keyword arguments.

        Returns:
            Node: A new Node instance.
        """
        return cls(
            type=NodeType.AGENT,
            agent=agent,
            id=getattr(agent, "agent_name", None),
            **kwargs,
        )


class Edge:
    """
    Represents an edge in a graph workflow.

    Attributes:
        source (str): The ID of the source node.
        target (str): The ID of the target node.
        metadata (Dict[str, Any], optional): Additional metadata for the edge.
    """

    def __init__(
        self,
        source: str = None,
        target: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize an Edge.

        Args:
            source (str, optional): The ID of the source node.
            target (str, optional): The ID of the target node.
            metadata (Dict[str, Any], optional): Additional metadata for the edge.
        """
        self.source = source
        self.target = target
        self.metadata = metadata or {}

    @classmethod
    def from_nodes(
        cls,
        source_node: Union["Node", Agent, str],
        target_node: Union["Node", Agent, str],
        **kwargs: Any,
    ) -> "Edge":
        """
        Create an Edge from node objects or ids.

        Args:
            source_node (Union[Node, Agent, str]): Source node object or ID.
            target_node (Union[Node, Agent, str]): Target node object or ID.
            **kwargs: Additional keyword arguments.

        Returns:
            Edge: A new Edge instance.
        """
        # Handle source node: extract ID from Node, Agent, or use string directly
        if isinstance(source_node, Node):
            src = source_node.id
        elif hasattr(source_node, "agent_name"):
            # Agent object - extract agent_name
            src = getattr(source_node, "agent_name", None)
            if src is None:
                raise ValueError(
                    "Source agent does not have an agent_name attribute"
                )
        else:
            # Assume it's already a string ID
            src = source_node

        # Handle target node: extract ID from Node, Agent, or use string directly
        if isinstance(target_node, Node):
            tgt = target_node.id
        elif hasattr(target_node, "agent_name"):
            # Agent object - extract agent_name
            tgt = getattr(target_node, "agent_name", None)
            if tgt is None:
                raise ValueError(
                    "Target agent does not have an agent_name attribute"
                )
        else:
            # Assume it's already a string ID
            tgt = target_node

        # Put all kwargs into metadata dict
        metadata = kwargs if kwargs else None
        return cls(source=src, target=tgt, metadata=metadata)


class GraphWorkflow:
    """
    Represents a workflow graph where each node is an agent.

    Attributes:
        nodes (Dict[str, Node]): A dictionary of nodes in the graph, where the key is the node ID and the value is the Node object.
        edges (List[Edge]): A list of edges in the graph, where each edge is represented by an Edge object.
        entry_points (List[str]): A list of node IDs that serve as entry points to the graph.
        end_points (List[str]): A list of node IDs that serve as end points of the graph.
        graph_backend (GraphBackend): A graph backend object (NetworkX or Rustworkx) representing the workflow graph.
        task (str): The task to be executed by the workflow.
        _compiled (bool): Whether the graph has been compiled for optimization.
        _sorted_layers (List[List[str]]): Pre-computed topological layers for faster execution.
        _max_workers (int): Pre-computed max workers for thread pool.
        verbose (bool): Whether to enable verbose logging.
    """

    def __init__(
        self,
        id: Optional[str] = str(uuid.uuid4()),
        name: Optional[str] = "Graph-Workflow-01",
        description: Optional[
            str
        ] = "A customizable workflow system for orchestrating and coordinating multiple agents.",
        nodes: Optional[Dict[str, Node]] = None,
        edges: Optional[List[Edge]] = None,
        entry_points: Optional[List[str]] = None,
        end_points: Optional[List[str]] = None,
        max_loops: int = 1,
        task: Optional[str] = None,
        auto_compile: bool = True,
        verbose: bool = False,
        backend: str = "networkx",
    ):
        self.id = id
        self.verbose = verbose

        if self.verbose:
            logger.info("Initializing GraphWorkflow")
            logger.debug(
                f"GraphWorkflow parameters: nodes={len(nodes) if nodes else 0}, edges={len(edges) if edges else 0}, max_loops={max_loops}, auto_compile={auto_compile}, backend={backend}"
            )

        self.nodes = nodes or {}
        self.edges = edges or []
        self.entry_points = entry_points or []
        self.end_points = end_points or []

        # Initialize graph backend
        if backend.lower() == "rustworkx":
            if not RUSTWORKX_AVAILABLE:
                logger.warning(
                    "rustworkx is not available, falling back to networkx. Install with: pip install rustworkx"
                )
                self.graph_backend = NetworkXBackend()
            else:
                self.graph_backend = RustworkxBackend()
                if self.verbose:
                    logger.info("Using rustworkx backend")
        else:
            self.graph_backend = NetworkXBackend()
            if self.verbose:
                logger.info("Using networkx backend")

        self.max_loops = max_loops
        self.task = task
        self.name = name
        self.description = description
        self.auto_compile = auto_compile

        # Private optimization attributes
        self._compiled = False
        self._sorted_layers = []
        self._max_workers = max(1, int(get_cpu_cores() * 0.95))
        self._compilation_timestamp = None

        if self.verbose:
            logger.debug(
                f"GraphWorkflow max_workers set to: {self._max_workers}"
            )

        self.conversation = Conversation()

        # Rebuild the graph from nodes and edges if provided
        if self.nodes:
            backend_name = (
                "rustworkx"
                if isinstance(self.graph_backend, RustworkxBackend)
                else "networkx"
            )
            if self.verbose:
                logger.info(
                    f"Adding {len(self.nodes)} nodes to {backend_name} graph"
                )

            for node_id, node in self.nodes.items():
                self.graph_backend.add_node(
                    node_id,
                    type=node.type,
                    agent=node.agent,
                    **(node.metadata or {}),
                )
                if self.verbose:
                    logger.debug(
                        f"Added node: {node_id} (type: {node.type})"
                    )

        if self.edges:
            backend_name = (
                "rustworkx"
                if isinstance(self.graph_backend, RustworkxBackend)
                else "networkx"
            )
            if self.verbose:
                logger.info(
                    f"Adding {len(self.edges)} edges to {backend_name} graph"
                )

            valid_edges = 0
            for edge in self.edges:
                if (
                    edge.source in self.nodes
                    and edge.target in self.nodes
                ):
                    self.graph_backend.add_edge(
                        edge.source,
                        edge.target,
                        **(edge.metadata or {}),
                    )
                    valid_edges += 1
                    if self.verbose:
                        logger.debug(
                            f"Added edge: {edge.source} -> {edge.target}"
                        )
                else:
                    logger.warning(
                        f"Skipping invalid edge: {edge.source} -> {edge.target} (nodes not found)"
                    )

            if self.verbose:
                logger.info(
                    f"Successfully added {valid_edges} valid edges"
                )

        # Auto-compile if requested and graph has nodes
        if self.auto_compile and self.nodes:
            if self.verbose:
                logger.info("Auto-compiling GraphWorkflow")
            self.compile()

        if self.verbose:
            logger.success(
                "GraphWorkflow initialization completed successfully"
            )

    def _invalidate_compilation(self) -> None:
        """
        Invalidate compiled optimizations when graph structure changes.
        Forces recompilation on next run to ensure cache coherency.
        """
        if self.verbose:
            logger.debug(
                "Invalidating compilation cache due to graph structure change"
            )

        self._compiled = False
        self._sorted_layers = []
        self._compilation_timestamp = None

        # Clear predecessors cache when graph structure changes
        if hasattr(self, "_predecessors_cache"):
            self._predecessors_cache = {}
            if self.verbose:
                logger.debug("Cleared predecessors cache")

    def compile(self) -> None:
        """
        Pre-compute expensive operations for faster execution.
        Call this after building the graph structure.
        Results are cached to avoid recompilation in multi-loop scenarios.
        """
        # Skip compilation if already compiled and graph structure hasn't changed
        if self._compiled:
            if self.verbose:
                logger.debug(
                    "GraphWorkflow already compiled, skipping recompilation"
                )
            return

        if self.verbose:
            logger.info("Starting GraphWorkflow compilation")

        compile_start_time = time.time()

        try:
            if not self.entry_points:
                if self.verbose:
                    logger.debug("Auto-setting entry points")
                self.auto_set_entry_points()

            if not self.end_points:
                if self.verbose:
                    logger.debug("Auto-setting end points")
                self.auto_set_end_points()

            if self.verbose:
                logger.debug(f"Entry points: {self.entry_points}")
                logger.debug(f"End points: {self.end_points}")

            # Pre-compute topological layers for efficient execution
            if self.verbose:
                logger.debug("Computing topological layers")

            sorted_layers = (
                self.graph_backend.topological_generations()
            )
            self._sorted_layers = sorted_layers

            # Cache compilation timestamp for debugging
            self._compilation_timestamp = time.time()
            self._compiled = True

            compile_time = time.time() - compile_start_time

            # Log compilation caching info for multi-loop scenarios
            cache_msg = ""
            if self.max_loops > 1:
                cache_msg = f" (cached for {self.max_loops} loops)"

            logger.info(
                f"GraphWorkflow compiled successfully: {len(self._sorted_layers)} layers, {len(self.nodes)} nodes (took {compile_time:.3f}s){cache_msg}"
            )

            if self.verbose:
                for i, layer in enumerate(self._sorted_layers):
                    logger.debug(f"Layer {i}: {layer}")

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow compilation: {e}"
            )
            raise e

    def add_node(self, agent: Agent, **kwargs: Any) -> None:
        """
        Adds an agent node to the workflow graph.

        Args:
            agent (Agent): The agent to add as a node.
            **kwargs: Additional keyword arguments for the node.
        """
        if self.verbose:
            logger.debug(
                f"Adding node for agent: {getattr(agent, 'agent_name', 'unnamed')}"
            )

        try:
            node = Node.from_agent(agent, **kwargs)

            if node.id in self.nodes:
                error_msg = f"Node with id {node.id} already exists in GraphWorkflow"
                logger.error(error_msg)
                raise ValueError(error_msg)

            self.nodes[node.id] = node
            self.graph_backend.add_node(
                node.id,
                type=node.type,
                agent=node.agent,
                **(node.metadata or {}),
            )
            self._invalidate_compilation()

            if self.verbose:
                logger.success(f"Successfully added node: {node.id}")

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.add_node for agent {getattr(agent, 'agent_name', 'unnamed')}: {e}"
            )
            raise e

    def add_nodes(
        self, agents: List[Agent], batch_size: int = 10, **kwargs: Any
    ) -> None:
        """
        Add multiple agents to the workflow graph concurrently in batches.

        Args:
            agents (List[Agent]): List of agents to add.
            batch_size (int): Number of agents to add concurrently in a batch. Defaults to 8.
            **kwargs: Additional keyword arguments for each node addition.
        """

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers
            ) as executor:
                # Process agents in batches
                for i in range(0, len(agents), batch_size):
                    batch = agents[i : i + batch_size]
                    futures = [
                        executor.submit(
                            self.add_node, agent, **kwargs
                        )
                        for agent in batch
                    ]
                    # Ensure all nodes in batch are added before next batch
                    for future in concurrent.futures.as_completed(
                        futures
                    ):
                        future.result()
        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.add_nodes for agents {agents}: {e} Traceback: {traceback.format_exc()}"
            )
            raise e

    def add_edge(
        self,
        edge_or_source: Union[Edge, Node, Agent, str],
        target: Optional[Union[Node, Agent, str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add an edge by Edge object or by passing node objects/ids.

        Args:
            edge_or_source (Union[Edge, Node, Agent, str]): Either an Edge object or the source node/id.
            target (Optional[Union[Node, Agent, str]]): Target node/id (required if edge_or_source is not an Edge).
            **kwargs: Additional keyword arguments for the edge.
        """
        try:
            if isinstance(edge_or_source, Edge):
                edge = edge_or_source
                if self.verbose:
                    logger.debug(
                        f"Adding edge object: {edge.source} -> {edge.target}"
                    )
            else:
                edge = Edge.from_nodes(
                    edge_or_source, target, **kwargs
                )
                if self.verbose:
                    logger.debug(
                        f"Creating and adding edge: {edge.source} -> {edge.target}"
                    )

            # Validate nodes exist
            if edge.source not in self.nodes:
                error_msg = f"Source node '{edge.source}' does not exist in GraphWorkflow"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if edge.target not in self.nodes:
                error_msg = f"Target node '{edge.target}' does not exist in GraphWorkflow"
                logger.error(error_msg)
                raise ValueError(error_msg)

            self.edges.append(edge)
            self.graph_backend.add_edge(
                edge.source, edge.target, **(edge.metadata or {})
            )
            self._invalidate_compilation()

            if self.verbose:
                logger.success(
                    f"Successfully added edge: {edge.source} -> {edge.target}"
                )

        except Exception as e:
            logger.exception(f"Error in GraphWorkflow.add_edge: {e}")
            raise e

    def add_edges_from_source(
        self,
        source: Union[Node, Agent, str],
        targets: List[Union[Node, Agent, str]],
        **kwargs: Any,
    ) -> List[Edge]:
        """
        Add multiple edges from a single source to multiple targets for parallel processing.
        This creates a "fan-out" pattern where the source agent's output is distributed
        to all target agents simultaneously.

        Args:
            source (Union[Node, Agent, str]): Source node/id that will send output to multiple targets.
            targets (List[Union[Node, Agent, str]]): List of target node/ids that will receive the source output in parallel.
            **kwargs: Additional keyword arguments for all edges.

        Returns:
            List[Edge]: List of created Edge objects.

        Example:
            # One agent's output goes to three specialists in parallel
            workflow.add_edges_from_source(
                "DataCollector",
                ["TechnicalAnalyst", "FundamentalAnalyst", "SentimentAnalyst"]
            )
        """
        if self.verbose:
            logger.info(
                f"Adding fan-out edges from {source} to {len(targets)} targets: {targets}"
            )

        created_edges = []

        try:
            for target in targets:
                edge = Edge.from_nodes(source, target, **kwargs)

                # Validate nodes exist
                if edge.source not in self.nodes:
                    error_msg = f"Source node '{edge.source}' does not exist in GraphWorkflow"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                if edge.target not in self.nodes:
                    error_msg = f"Target node '{edge.target}' does not exist in GraphWorkflow"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                self.edges.append(edge)
                self.graph_backend.add_edge(
                    edge.source, edge.target, **(edge.metadata or {})
                )
                created_edges.append(edge)

                if self.verbose:
                    logger.debug(
                        f"Added fan-out edge: {edge.source} -> {edge.target}"
                    )

            self._invalidate_compilation()

            if self.verbose:
                logger.success(
                    f"Successfully added {len(created_edges)} fan-out edges from {source}"
                )

            return created_edges

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.add_edges_from_source: {e}"
            )
            raise e

    def add_edges_to_target(
        self,
        sources: List[Union[Node, Agent, str]],
        target: Union[Node, Agent, str],
        **kwargs: Any,
    ) -> List[Edge]:
        """
        Add multiple edges from multiple sources to a single target for convergence processing.
        This creates a "fan-in" pattern where multiple agents' outputs converge to a single target.

        Args:
            sources (List[Union[Node, Agent, str]]): List of source node/ids that will send output to the target.
            target (Union[Node, Agent, str]): Target node/id that will receive all source outputs.
            **kwargs: Additional keyword arguments for all edges.

        Returns:
            List[Edge]: List of created Edge objects.

        Example:
            # Multiple specialists send results to a synthesis agent
            workflow.add_edges_to_target(
                ["TechnicalAnalyst", "FundamentalAnalyst", "SentimentAnalyst"],
                "SynthesisAgent"
            )
        """
        if self.verbose:
            logger.info(
                f"Adding fan-in edges from {len(sources)} sources to {target}: {sources}"
            )

        created_edges = []

        try:
            for source in sources:
                edge = Edge.from_nodes(source, target, **kwargs)

                # Validate nodes exist
                if edge.source not in self.nodes:
                    error_msg = f"Source node '{edge.source}' does not exist in GraphWorkflow"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                if edge.target not in self.nodes:
                    error_msg = f"Target node '{edge.target}' does not exist in GraphWorkflow"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                self.edges.append(edge)
                self.graph_backend.add_edge(
                    edge.source, edge.target, **(edge.metadata or {})
                )
                created_edges.append(edge)

                if self.verbose:
                    logger.debug(
                        f"Added fan-in edge: {edge.source} -> {edge.target}"
                    )

            self._invalidate_compilation()

            if self.verbose:
                logger.success(
                    f"Successfully added {len(created_edges)} fan-in edges to {target}"
                )

            return created_edges

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.add_edges_to_target: {e}"
            )
            raise e

    def add_parallel_chain(
        self,
        sources: List[Union[Node, Agent, str]],
        targets: List[Union[Node, Agent, str]],
        **kwargs: Any,
    ) -> List[Edge]:
        """
        Create a parallel processing chain where multiple sources connect to multiple targets.
        This creates a full mesh connection pattern for maximum parallel processing.

        Args:
            sources (List[Union[Node, Agent, str]]): List of source node/ids.
            targets (List[Union[Node, Agent, str]]): List of target node/ids.
            **kwargs: Additional keyword arguments for all edges.

        Returns:
            List[Edge]: List of created Edge objects.

        Example:
            # Multiple data collectors feed multiple analysts
            workflow.add_parallel_chain(
                ["DataCollector1", "DataCollector2"],
                ["Analyst1", "Analyst2", "Analyst3"]
            )
        """
        if self.verbose:
            logger.info(
                f"Creating parallel chain: {len(sources)} sources -> {len(targets)} targets"
            )

        created_edges = []

        try:
            for source in sources:
                for target in targets:
                    edge = Edge.from_nodes(source, target, **kwargs)

                    # Validate nodes exist
                    if edge.source not in self.nodes:
                        error_msg = f"Source node '{edge.source}' does not exist in GraphWorkflow"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    if edge.target not in self.nodes:
                        error_msg = f"Target node '{edge.target}' does not exist in GraphWorkflow"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    self.edges.append(edge)
                    self.graph_backend.add_edge(
                        edge.source,
                        edge.target,
                        **(edge.metadata or {}),
                    )
                    created_edges.append(edge)

                    if self.verbose:
                        logger.debug(
                            f"Added parallel edge: {edge.source} -> {edge.target}"
                        )

            self._invalidate_compilation()

            if self.verbose:
                logger.success(
                    f"Successfully created parallel chain with {len(created_edges)} edges"
                )

            return created_edges

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.add_parallel_chain: {e}"
            )
            raise e

    def set_entry_points(self, entry_points: List[str]) -> None:
        """
        Set the entry points for the workflow.

        Args:
            entry_points (List[str]): List of node IDs to serve as entry points.
        """
        if self.verbose:
            logger.debug(f"Setting entry points: {entry_points}")

        try:
            for node_id in entry_points:
                if node_id not in self.nodes:
                    error_msg = f"Entry point node '{node_id}' does not exist in GraphWorkflow"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            self.entry_points = entry_points
            self._invalidate_compilation()

            if self.verbose:
                logger.success(
                    f"Successfully set entry points: {entry_points}"
                )

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.set_entry_points: {e}"
            )
            raise e

    def set_end_points(self, end_points: List[str]) -> None:
        """
        Set the end points for the workflow.

        Args:
            end_points (List[str]): List of node IDs to serve as end points.
        """
        if self.verbose:
            logger.debug(f"Setting end points: {end_points}")

        try:
            for node_id in end_points:
                if node_id not in self.nodes:
                    error_msg = f"End point node '{node_id}' does not exist in GraphWorkflow"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            self.end_points = end_points
            self._invalidate_compilation()

            if self.verbose:
                logger.success(
                    f"Successfully set end points: {end_points}"
                )

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.set_end_points: {e}"
            )
            raise e

    @classmethod
    def from_spec(
        cls,
        agents: List[Union[Agent, Node]],
        edges: List[Union[Edge, Tuple[Any, Any]]],
        entry_points: Optional[List[str]] = None,
        end_points: Optional[List[str]] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> "GraphWorkflow":
        """
        Construct a workflow from a list of agents and connections.

        Args:
            agents (List[Union[Agent, Node]]): List of agents or Node objects.
            edges (List[Union[Edge, Tuple[Any, Any]]]): List of edges or edge tuples.
            entry_points (Optional[List[str]]): List of entry point node IDs.
            end_points (Optional[List[str]]): List of end point node IDs.
            task (Optional[str]): Task to be executed by the workflow.
            **kwargs: Additional keyword arguments.

        Returns:
            GraphWorkflow: A new GraphWorkflow instance.
        """
        verbose = kwargs.get("verbose", False)

        if verbose:
            logger.info(
                f"Creating GraphWorkflow from spec with {len(agents)} agents and {len(edges)} edges"
            )

        try:
            wf = cls(task=task, **kwargs)
            node_objs = []

            for i, agent in enumerate(agents):
                if isinstance(agent, Node):
                    node_objs.append(agent)
                    if verbose:
                        logger.debug(
                            f"Added Node object {i+1}/{len(agents)}: {agent.id}"
                        )
                elif hasattr(agent, "agent_name"):
                    node_obj = Node.from_agent(agent)
                    node_objs.append(node_obj)
                    if verbose:
                        logger.debug(
                            f"Created Node {i+1}/{len(agents)} from agent: {node_obj.id}"
                        )
                else:
                    error_msg = f"Unknown node type at index {i}: {type(agent)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            for node in node_objs:
                wf.add_node(node.agent)

            for i, e in enumerate(edges):
                if isinstance(e, Edge):
                    wf.add_edge(e)
                    if verbose:
                        logger.debug(
                            f"Added Edge object {i+1}/{len(edges)}: {e.source} -> {e.target}"
                        )
                elif isinstance(e, (tuple, list)) and len(e) >= 2:
                    # Support various edge formats:
                    # - (source, target) - single edge
                    # - (source, [target1, target2]) - fan-out from source
                    # - ([source1, source2], target) - fan-in to target
                    # - ([source1, source2], [target1, target2]) - parallel chain
                    source, target = e[0], e[1]

                    if isinstance(
                        source, (list, tuple)
                    ) and isinstance(target, (list, tuple)):
                        # Parallel chain: multiple sources to multiple targets
                        wf.add_parallel_chain(source, target)
                        if verbose:
                            logger.debug(
                                f"Added parallel chain {i+1}/{len(edges)}: {len(source)} sources -> {len(target)} targets"
                            )
                    elif isinstance(target, (list, tuple)):
                        # Fan-out: single source to multiple targets
                        wf.add_edges_from_source(source, target)
                        if verbose:
                            logger.debug(
                                f"Added fan-out {i+1}/{len(edges)}: {source} -> {len(target)} targets"
                            )
                    elif isinstance(source, (list, tuple)):
                        # Fan-in: multiple sources to single target
                        wf.add_edges_to_target(source, target)
                        if verbose:
                            logger.debug(
                                f"Added fan-in {i+1}/{len(edges)}: {len(source)} sources -> {target}"
                            )
                    else:
                        # Simple edge: single source to single target
                        wf.add_edge(source, target)
                        if verbose:
                            logger.debug(
                                f"Added edge {i+1}/{len(edges)}: {source} -> {target}"
                            )
                else:
                    error_msg = (
                        f"Unknown edge type at index {i}: {type(e)}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            if entry_points:
                wf.set_entry_points(entry_points)
            else:
                wf.auto_set_entry_points()

            if end_points:
                wf.set_end_points(end_points)
            else:
                wf.auto_set_end_points()

            # Auto-compile after construction
            wf.compile()

            if verbose:
                logger.success(
                    "Successfully created GraphWorkflow from spec"
                )

            return wf

        except Exception as e:
            logger.exception(f"Error in GraphWorkflow.from_spec: {e}")
            raise e

    def auto_set_entry_points(self) -> None:
        """
        Automatically set entry points to nodes with no incoming edges.
        """
        if self.verbose:
            logger.debug("Auto-setting entry points")

        try:
            self.entry_points = [
                n
                for n in self.nodes
                if self.graph_backend.in_degree(n) == 0
            ]

            if self.verbose:
                logger.info(
                    f"Auto-set entry points: {self.entry_points}"
                )

            if not self.entry_points and self.nodes:
                logger.warning(
                    "No entry points found - all nodes have incoming edges (possible cycle)"
                )

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.auto_set_entry_points: {e}"
            )
            raise e

    def auto_set_end_points(self) -> None:
        """
        Automatically set end points to nodes with no outgoing edges.
        """
        if self.verbose:
            logger.debug("Auto-setting end points")

        try:
            self.end_points = [
                n
                for n in self.nodes
                if self.graph_backend.out_degree(n) == 0
            ]

            if self.verbose:
                logger.info(f"Auto-set end points: {self.end_points}")

            if not self.end_points and self.nodes:
                logger.warning(
                    "No end points found - all nodes have outgoing edges (possible cycle)"
                )

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.auto_set_end_points: {e}"
            )
            raise e

    def _get_predecessors(self, node_id: str) -> Tuple[str, ...]:
        """
        Cached predecessor lookup for faster repeated access.

        Args:
            node_id (str): The node ID to get predecessors for.

        Returns:
            Tuple[str, ...]: Tuple of predecessor node IDs.
        """
        # Use instance-level caching instead of @lru_cache to avoid hashing issues
        if not hasattr(self, "_predecessors_cache"):
            self._predecessors_cache = {}

        if node_id not in self._predecessors_cache:
            self._predecessors_cache[node_id] = tuple(
                self.graph_backend.predecessors(node_id)
            )

        return self._predecessors_cache[node_id]

    def _build_prompt(
        self,
        node_id: str,
        task: str,
        prev_outputs: Dict[str, Any],
        layer_idx: int,
    ) -> str:
        """
        Optimized prompt building with minimal string operations.

        Args:
            node_id (str): The node ID to build a prompt for.
            task (str): The main task.
            prev_outputs (Dict[str, Any]): Previous outputs from predecessor nodes.
            layer_idx (int): The current layer index.

        Returns:
            str: The built prompt.
        """
        if self.verbose:
            logger.debug(
                f"Building prompt for node {node_id} (layer {layer_idx})"
            )

        try:
            preds = self._get_predecessors(node_id)
            pred_outputs = [
                prev_outputs.get(pred)
                for pred in preds
                if pred in prev_outputs
            ]

            if pred_outputs and layer_idx > 0:
                # Use list comprehension and join for faster string building
                predecessor_parts = [
                    f"Output from {pred}:\n{out}"
                    for pred, out in zip(preds, pred_outputs)
                    if out is not None
                ]
                predecessor_context = "\n\n".join(predecessor_parts)

                prompt = (
                    f"Original Task: {task}\n\n"
                    f"Previous Agent Outputs:\n{predecessor_context}\n\n"
                    f"Instructions: Please carefully review the work done by your predecessor agents above. "
                    f"Acknowledge their contributions, verify their findings, and build upon their work. "
                    f"If you agree with their analysis, say so and expand on it. "
                    f"If you disagree or find gaps, explain why and provide corrections or improvements. "
                    f"Your goal is to collaborate and create a comprehensive response that builds on all previous work."
                )
            else:
                prompt = (
                    f"{task}\n\n"
                    f"You are starting the workflow analysis. Please provide your best comprehensive response to this task."
                )

            if self.verbose:
                logger.debug(
                    f"Built prompt for node {node_id} ({len(prompt)} characters)"
                )

            return prompt

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow._build_prompt for node {node_id}: {e}"
            )
            raise e

    async def arun(
        self,
        task: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Async version of run for better performance with I/O bound operations.

        Args:
            task (Optional[str]): Task to execute. Uses self.task if not provided.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Execution results from all nodes.
        """
        if self.verbose:
            logger.info("Starting async GraphWorkflow execution")

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.run, task, *args, **kwargs
            )

            if self.verbose:
                logger.success(
                    "Async GraphWorkflow execution completed"
                )

            return result

        except Exception as e:
            logger.exception(f"Error in GraphWorkflow.arun: {e}")
            raise e

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the workflow graph with optimized parallel agent execution.

        Args:
            task (Optional[str]): Task to execute. Uses self.task if not provided.
            img (Optional[str]): Optional image path for multimodal tasks.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Execution results from all nodes.
        """
        run_start_time = time.time()

        if task is not None:
            self.task = task
        else:
            task = self.task

        if self.verbose:
            logger.info(
                f"Starting GraphWorkflow execution with task: {task[:100]}{'...' if len(str(task)) > 100 else ''}"
            )
            logger.debug(
                f"Execution parameters: max_loops={self.max_loops}, max_workers={self._max_workers}"
            )

        # Ensure compilation is done once and cached for multi-loop execution
        compilation_needed = not self._compiled
        if compilation_needed:
            if self.verbose:
                compile_msg = "Graph not compiled, compiling now"
                if self.max_loops > 1:
                    compile_msg += f" (will be cached for {self.max_loops} loops)"
                logger.info(compile_msg)
            self.compile()
        elif self.max_loops > 1 and self.verbose:
            logger.debug(
                f"Using cached compilation for {self.max_loops} loops (compiled at {getattr(self, '_compilation_timestamp', 'unknown time')})"
            )

        try:
            loop = 0
            while loop < self.max_loops:
                loop_start_time = time.time()

                if self.verbose:
                    cache_status = (
                        " (using cached structure)"
                        if loop > 0 or not compilation_needed
                        else ""
                    )
                    logger.info(
                        f"Starting execution loop {loop + 1}/{self.max_loops}{cache_status}"
                    )

                execution_results = {}
                prev_outputs = {}

                for layer_idx, layer in enumerate(
                    self._sorted_layers
                ):
                    layer_start_time = time.time()

                    if self.verbose:
                        logger.info(
                            f"Executing layer {layer_idx + 1}/{len(self._sorted_layers)} with {len(layer)} nodes: {layer}"
                        )

                    # Pre-build all prompts for this layer
                    layer_data = []
                    for node_id in layer:
                        try:
                            prompt = self._build_prompt(
                                node_id, task, prev_outputs, layer_idx
                            )
                            layer_data.append(
                                (
                                    node_id,
                                    self.nodes[node_id].agent,
                                    prompt,
                                )
                            )
                        except Exception as e:
                            logger.exception(
                                f"Error building prompt for node {node_id}: {e}"
                            )
                            # Continue with empty prompt as fallback
                            layer_data.append(
                                (
                                    node_id,
                                    self.nodes[node_id].agent,
                                    f"Error building prompt: {e}",
                                )
                            )

                    # Execute all agents in this layer in parallel
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(self._max_workers, len(layer))
                    ) as executor:

                        if self.verbose:
                            logger.debug(
                                f"Created thread pool with {min(self._max_workers, len(layer))} workers for layer {layer_idx + 1}"
                            )

                        future_to_data = {}

                        # Submit all tasks
                        for node_id, agent, prompt in layer_data:
                            try:
                                future = executor.submit(
                                    agent.run,
                                    prompt,
                                    img,
                                    *args,
                                    **kwargs,
                                )
                                future_to_data[future] = (
                                    node_id,
                                    agent,
                                )

                                if self.verbose:
                                    logger.debug(
                                        f"Submitted execution task for agent: {getattr(agent, 'agent_name', node_id)}"
                                    )

                            except Exception as e:
                                logger.exception(
                                    f"Error submitting task for agent {getattr(agent, 'agent_name', node_id)}: {e}"
                                )
                                # Add error result directly
                                error_output = f"[ERROR] Failed to submit task: {e}"
                                prev_outputs[node_id] = error_output
                                execution_results[node_id] = (
                                    error_output
                                )

                        # Collect results as they complete
                        completed_count = 0
                        for future in concurrent.futures.as_completed(
                            future_to_data
                        ):
                            node_id, agent = future_to_data[future]
                            agent_name = getattr(
                                agent, "agent_name", node_id
                            )

                            try:
                                agent_start_time = time.time()
                                output = future.result()
                                agent_execution_time = (
                                    time.time() - agent_start_time
                                )

                                completed_count += 1

                                if self.verbose:
                                    logger.success(
                                        f"Agent {agent_name} completed successfully ({completed_count}/{len(layer_data)}) in {agent_execution_time:.3f}s"
                                    )

                            except Exception as e:
                                output = f"[ERROR] Agent {agent_name} failed: {e}"
                                logger.exception(
                                    f"Error in GraphWorkflow agent execution for {agent_name}: {e}"
                                )

                            prev_outputs[node_id] = output
                            execution_results[node_id] = output

                            # Add to conversation (this could be optimized further by batching)
                            try:
                                self.conversation.add(
                                    role=agent_name,
                                    content=output,
                                )

                                if self.verbose:
                                    logger.debug(
                                        f"Added output to conversation for agent: {agent_name}"
                                    )

                            except Exception as e:
                                logger.exception(
                                    f"Error adding output to conversation for agent {agent_name}: {e}"
                                )

                    layer_execution_time = (
                        time.time() - layer_start_time
                    )

                    if self.verbose:
                        logger.success(
                            f"Layer {layer_idx + 1} completed in {layer_execution_time:.3f}s"
                        )

                loop_execution_time = time.time() - loop_start_time
                loop += 1

                if self.verbose:
                    logger.success(
                        f"Loop {loop}/{self.max_loops} completed in {loop_execution_time:.3f}s"
                    )

                # For now, we still return after the first loop
                # This maintains backward compatibility
                total_execution_time = time.time() - run_start_time

                logger.info(
                    f"GraphWorkflow execution completed: {len(execution_results)} agents executed in {total_execution_time:.3f}s"
                )

                if self.verbose:
                    logger.debug(
                        f"Final execution results: {list(execution_results.keys())}"
                    )

                return execution_results

        except Exception as e:
            total_time = time.time() - run_start_time
            logger.exception(
                f"Error in GraphWorkflow.run after {total_time:.3f}s: {e}"
            )
            raise e

    def visualize(
        self,
        format: str = "png",
        view: bool = True,
        engine: str = "dot",
        show_summary: bool = False,
    ) -> str:
        """
        Visualize the workflow graph using Graphviz with enhanced parallel pattern detection.

        Args:
            format (str): Output format ('png', 'svg', 'pdf', 'dot'). Defaults to 'png'.
            view (bool): Whether to open the visualization after creation. Defaults to True.
            engine (str): Graphviz layout engine ('dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo'). Defaults to 'dot'.
            show_summary (bool): Whether to print parallel processing summary. Defaults to False.

        Returns:
            str: Path to the generated visualization file.

        Raises:
            ImportError: If graphviz is not installed.
            Exception: If visualization generation fails.
        """
        output_path = f"{self.name}_visualization_{str(uuid.uuid4())}"

        if not GRAPHVIZ_AVAILABLE:
            error_msg = "Graphviz is not installed. Install it with: pip install graphviz"
            logger.error(error_msg)
            raise ImportError(error_msg)

        if self.verbose:
            logger.debug(
                f"Visualizing GraphWorkflow with Graphviz (format={format}, engine={engine})"
            )

        try:
            # Create Graphviz digraph
            dot = graphviz.Digraph(
                name=f"GraphWorkflow_{self.name or 'Unnamed'}",
                comment=f"GraphWorkflow: {self.description or 'No description'}",
                engine=engine,
                format=format,
            )

            # Set graph attributes for better visualization
            dot.attr(rankdir="TB")  # Top to bottom layout
            dot.attr(bgcolor="white")
            dot.attr(fontname="Arial")
            dot.attr(fontsize="12")
            dot.attr(labelloc="t")  # Title at top
            dot.attr(
                label=f'GraphWorkflow: {self.name or "Unnamed"}\\n{len(self.nodes)} Agents, {len(self.edges)} Connections'
            )

            # Set default node attributes
            dot.attr(
                "node",
                shape="box",
                style="rounded,filled",
                fontname="Arial",
                fontsize="10",
                margin="0.1,0.05",
            )

            # Set default edge attributes
            dot.attr(
                "edge",
                fontname="Arial",
                fontsize="8",
                arrowsize="0.8",
            )

            # Analyze patterns for enhanced visualization
            fan_out_nodes = {}  # source -> [targets]
            fan_in_nodes = {}  # target -> [sources]

            for edge in self.edges:
                # Track fan-out patterns
                if edge.source not in fan_out_nodes:
                    fan_out_nodes[edge.source] = []
                fan_out_nodes[edge.source].append(edge.target)

                # Track fan-in patterns
                if edge.target not in fan_in_nodes:
                    fan_in_nodes[edge.target] = []
                fan_in_nodes[edge.target].append(edge.source)

            # Add nodes with styling based on their role
            for node_id, node in self.nodes.items():
                agent_name = getattr(
                    node.agent, "agent_name", node_id
                )

                # Determine node color and style based on role
                is_entry = node_id in self.entry_points
                is_exit = node_id in self.end_points
                is_fan_out = len(fan_out_nodes.get(node_id, [])) > 1
                is_fan_in = len(fan_in_nodes.get(node_id, [])) > 1

                # Choose colors based on node characteristics
                if is_entry:
                    fillcolor = (
                        "#E8F5E8"  # Light green for entry points
                    )
                    color = "#4CAF50"  # Green border
                elif is_exit:
                    fillcolor = (
                        "#F3E5F5"  # Light purple for end points
                    )
                    color = "#9C27B0"  # Purple border
                elif is_fan_out:
                    fillcolor = (
                        "#E3F2FD"  # Light blue for fan-out nodes
                    )
                    color = "#2196F3"  # Blue border
                elif is_fan_in:
                    fillcolor = (
                        "#FFF3E0"  # Light orange for fan-in nodes
                    )
                    color = "#FF9800"  # Orange border
                else:
                    fillcolor = (
                        "#F5F5F5"  # Light gray for regular nodes
                    )
                    color = "#757575"  # Gray border

                # Create node label with agent info
                label = f"{agent_name}"
                if is_entry:
                    label += "\\n(Entry)"
                if is_exit:
                    label += "\\n(Exit)"
                if is_fan_out:
                    label += (
                        f"\\n(Fan-out: {len(fan_out_nodes[node_id])})"
                    )
                if is_fan_in:
                    label += (
                        f"\\n(Fan-in: {len(fan_in_nodes[node_id])})"
                    )

                dot.node(
                    node_id,
                    label=label,
                    fillcolor=fillcolor,
                    color=color,
                    fontcolor="black",
                )

            # Add edges with styling based on pattern type

            for edge in self.edges:

                # Determine edge style based on pattern
                source_fan_out = (
                    len(fan_out_nodes.get(edge.source, [])) > 1
                )
                target_fan_in = (
                    len(fan_in_nodes.get(edge.target, [])) > 1
                )

                if source_fan_out and target_fan_in:
                    # Part of both fan-out and fan-in pattern
                    color = "#9C27B0"  # Purple
                    style = "bold"
                    penwidth = "2.0"
                elif source_fan_out:
                    # Part of fan-out pattern
                    color = "#2196F3"  # Blue
                    style = "solid"
                    penwidth = "1.5"
                elif target_fan_in:
                    # Part of fan-in pattern
                    color = "#FF9800"  # Orange
                    style = "solid"
                    penwidth = "1.5"
                else:
                    # Regular edge
                    color = "#757575"  # Gray
                    style = "solid"
                    penwidth = "1.0"

                # Add edge with metadata if available
                edge_label = ""
                if edge.metadata:
                    edge_label = str(edge.metadata)

                dot.edge(
                    edge.source,
                    edge.target,
                    label=edge_label,
                    color=color,
                    style=style,
                    penwidth=penwidth,
                )

            # Add subgraphs for better organization if compiled
            if self._compiled and len(self._sorted_layers) > 1:
                for layer_idx, layer in enumerate(
                    self._sorted_layers
                ):
                    with dot.subgraph(
                        name=f"cluster_layer_{layer_idx}"
                    ) as layer_graph:
                        layer_graph.attr(style="dashed")
                        layer_graph.attr(color="lightgray")
                        layer_graph.attr(
                            label=f"Layer {layer_idx + 1}"
                        )
                        layer_graph.attr(fontsize="10")

                        # Add invisible nodes to maintain layer structure
                        for node_id in layer:
                            layer_graph.node(node_id)

            # Generate output path
            if output_path is None:
                safe_name = "".join(
                    c if c.isalnum() or c in "-_" else "_"
                    for c in (self.name or "GraphWorkflow")
                )
                output_path = f"{safe_name}_visualization"

            # Render the graph
            output_file = dot.render(
                output_path, view=view, cleanup=True
            )

            # Show parallel processing summary
            if show_summary:
                fan_out_count = sum(
                    1
                    for targets in fan_out_nodes.values()
                    if len(targets) > 1
                )
                fan_in_count = sum(
                    1
                    for sources in fan_in_nodes.values()
                    if len(sources) > 1
                )
                total_parallel = len(
                    [
                        t
                        for targets in fan_out_nodes.values()
                        if len(targets) > 1
                        for t in targets
                    ]
                )

                print("\n" + "=" * 60)
                print("📊 GRAPHVIZ WORKFLOW VISUALIZATION")
                print("=" * 60)
                print(f"📁 Saved to: {output_file}")
                print(f"🤖 Total Agents: {len(self.nodes)}")
                print(f"🔗 Total Connections: {len(self.edges)}")
                if self._compiled:
                    print(
                        f"📚 Execution Layers: {len(self._sorted_layers)}"
                    )

                if fan_out_count > 0 or fan_in_count > 0:
                    print("\n⚡ Parallel Processing Patterns:")
                    if fan_out_count > 0:
                        print(
                            f"  🔀 Fan-out patterns: {fan_out_count}"
                        )
                    if fan_in_count > 0:
                        print(f"  🔀 Fan-in patterns: {fan_in_count}")
                    if total_parallel > 0:
                        print(
                            f"  ⚡ Parallel execution nodes: {total_parallel}"
                        )
                        efficiency = (
                            total_parallel / len(self.nodes)
                        ) * 100
                        print(
                            f"  🎯 Parallel efficiency: {efficiency:.1f}%"
                        )

                print("\n🎨 Legend:")
                print("  🟢 Green: Entry points")
                print("  🟣 Purple: Exit points")
                print("  🔵 Blue: Fan-out nodes")
                print("  🟠 Orange: Fan-in nodes")
                print("  ⚫ Gray: Regular nodes")

            if self.verbose:
                logger.success(
                    f"Graphviz visualization generated: {output_file}"
                )

            return output_file

        except Exception as e:
            logger.exception(f"Error in GraphWorkflow.visualize: {e}")
            raise e

    def visualize_simple(self) -> str:
        """
        Simple text-based visualization for environments without Graphviz.

        Returns:
            str: Text representation of the workflow.
        """
        if self.verbose:
            logger.debug("Generating simple text visualization")

        try:
            lines = []
            lines.append(f"GraphWorkflow: {self.name or 'Unnamed'}")
            lines.append(
                f"Description: {self.description or 'No description'}"
            )
            lines.append(
                f"Nodes: {len(self.nodes)}, Edges: {len(self.edges)}"
            )
            lines.append("")

            # Show nodes
            lines.append("🤖 Agents:")
            for node_id, node in self.nodes.items():
                agent_name = getattr(
                    node.agent, "agent_name", node_id
                )
                tags = []
                if node_id in self.entry_points:
                    tags.append("ENTRY")
                if node_id in self.end_points:
                    tags.append("EXIT")
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                lines.append(f"  - {agent_name}{tag_str}")

            lines.append("")

            # Show connections
            lines.append("🔗 Connections:")
            for edge in self.edges:
                lines.append(f"  {edge.source} → {edge.target}")

            # Show parallel patterns
            fan_out_nodes = {}
            fan_in_nodes = {}

            for edge in self.edges:
                if edge.source not in fan_out_nodes:
                    fan_out_nodes[edge.source] = []
                fan_out_nodes[edge.source].append(edge.target)

                if edge.target not in fan_in_nodes:
                    fan_in_nodes[edge.target] = []
                fan_in_nodes[edge.target].append(edge.source)

            fan_out_count = sum(
                1
                for targets in fan_out_nodes.values()
                if len(targets) > 1
            )
            fan_in_count = sum(
                1
                for sources in fan_in_nodes.values()
                if len(sources) > 1
            )

            if fan_out_count > 0 or fan_in_count > 0:
                lines.append("")
                lines.append("⚡ Parallel Patterns:")
                if fan_out_count > 0:
                    lines.append(
                        f"  🔀 Fan-out patterns: {fan_out_count}"
                    )
                if fan_in_count > 0:
                    lines.append(
                        f"  🔀 Fan-in patterns: {fan_in_count}"
                    )

            result = "\n".join(lines)
            print(result)
            return result

        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow.visualize_simple: {e}"
            )
            raise e

    def to_json(
        self,
        fast: bool = True,
        include_conversation: bool = False,
        include_runtime_state: bool = False,
    ) -> str:
        """
        Serialize the workflow to JSON with comprehensive metadata and configuration.

        Args:
            fast (bool): Whether to use fast JSON serialization. Defaults to True.
            include_conversation (bool): Whether to include conversation history. Defaults to False.
            include_runtime_state (bool): Whether to include runtime state like compilation info. Defaults to False.

        Returns:
            str: JSON representation of the workflow.
        """
        if self.verbose:
            logger.debug(
                f"Serializing GraphWorkflow to JSON (fast={fast}, include_conversation={include_conversation}, include_runtime_state={include_runtime_state})"
            )

        try:

            def node_to_dict(node: Node) -> Dict[str, Any]:
                node_data = {
                    "id": node.id,
                    "type": str(node.type),
                    "metadata": node.metadata,
                }

                # Serialize agent with enhanced error handling
                if hasattr(node.agent, "to_dict"):
                    try:
                        node_data["agent"] = node.agent.to_dict()
                    except Exception as e:
                        logger.warning(
                            f"Failed to serialize agent {node.id} to dict: {e}"
                        )
                        node_data["agent"] = {
                            "agent_name": getattr(
                                node.agent,
                                "agent_name",
                                str(node.agent),
                            ),
                            "serialization_error": str(e),
                            "agent_type": str(type(node.agent)),
                        }
                else:
                    node_data["agent"] = {
                        "agent_name": getattr(
                            node.agent, "agent_name", str(node.agent)
                        ),
                        "agent_type": str(type(node.agent)),
                        "serialization_method": "fallback_string",
                    }

                return node_data

            def edge_to_dict(edge: Edge) -> Dict[str, Any]:
                return {
                    "source": edge.source,
                    "target": edge.target,
                    "metadata": edge.metadata,
                }

            # Core workflow data
            data = {
                # Schema and versioning
                "schema_version": "1.0.0",
                "export_timestamp": time.time(),
                "export_date": time.strftime(
                    "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
                ),
                # Core identification
                "id": self.id,
                "name": self.name,
                "description": self.description,
                # Graph structure
                "nodes": [
                    node_to_dict(n) for n in self.nodes.values()
                ],
                "edges": [edge_to_dict(e) for e in self.edges],
                "entry_points": self.entry_points,
                "end_points": self.end_points,
                # Execution configuration
                "max_loops": self.max_loops,
                "auto_compile": self.auto_compile,
                "verbose": self.verbose,
                "task": self.task,
                # Performance configuration
                "max_workers": self._max_workers,
                # Graph metrics
                "metrics": {
                    "node_count": len(self.nodes),
                    "edge_count": len(self.edges),
                    "entry_point_count": len(self.entry_points),
                    "end_point_count": len(self.end_points),
                    "is_compiled": self._compiled,
                    "layer_count": (
                        len(self._sorted_layers)
                        if self._compiled
                        else None
                    ),
                },
            }

            # Optional conversation history
            if include_conversation and self.conversation:
                try:
                    if hasattr(self.conversation, "to_dict"):
                        data["conversation"] = (
                            self.conversation.to_dict()
                        )
                    elif hasattr(self.conversation, "history"):
                        data["conversation"] = {
                            "history": self.conversation.history,
                            "type": str(type(self.conversation)),
                        }
                    else:
                        data["conversation"] = {
                            "serialization_note": "Conversation object could not be serialized",
                            "type": str(type(self.conversation)),
                        }
                except Exception as e:
                    logger.warning(
                        f"Failed to serialize conversation: {e}"
                    )
                    data["conversation"] = {
                        "serialization_error": str(e)
                    }

            # Optional runtime state
            if include_runtime_state:
                data["runtime_state"] = {
                    "is_compiled": self._compiled,
                    "compilation_timestamp": self._compilation_timestamp,
                    "sorted_layers": (
                        self._sorted_layers
                        if self._compiled
                        else None
                    ),
                    "compilation_cache_valid": self._compiled,
                    "time_since_compilation": (
                        time.time() - self._compilation_timestamp
                        if self._compilation_timestamp
                        else None
                    ),
                }

            # Serialize to JSON
            if fast:
                result = json.dumps(data, indent=2, default=str)
            else:
                try:
                    from swarms.tools.json_utils import str_to_json

                    result = str_to_json(data, indent=2)
                except ImportError:
                    logger.warning(
                        "json_utils not available, falling back to standard json"
                    )
                    result = json.dumps(data, indent=2, default=str)

            if self.verbose:
                logger.success(
                    f"Successfully serialized GraphWorkflow to JSON ({len(result)} characters, {len(self.nodes)} nodes, {len(self.edges)} edges)"
                )

            return result

        except Exception as e:
            logger.exception(f"Error in GraphWorkflow.to_json: {e}")
            raise e

    @classmethod
    def from_json(
        cls,
        json_str: str,
        restore_runtime_state: bool = False,
    ) -> "GraphWorkflow":
        """
        Deserialize a workflow from JSON with comprehensive parameter support and backward compatibility.

        Args:
            json_str (str): JSON string representation of the workflow.
            restore_runtime_state (bool): Whether to restore runtime state like compilation info. Defaults to False.

        Returns:
            GraphWorkflow: A new GraphWorkflow instance with all parameters restored.
        """
        logger.debug(
            f"Deserializing GraphWorkflow from JSON ({len(json_str)} characters, restore_runtime_state={restore_runtime_state})"
        )

        try:
            data = json.loads(json_str)

            # Check for schema version and log compatibility info
            schema_version = data.get("schema_version", "legacy")
            export_date = data.get("export_date", "unknown")

            if schema_version != "legacy":
                logger.info(
                    f"Loading GraphWorkflow schema version {schema_version} exported on {export_date}"
                )
            else:
                logger.info("Loading legacy GraphWorkflow format")

            # Reconstruct nodes with enhanced agent handling
            nodes = []
            for n in data["nodes"]:
                try:
                    # Handle different agent serialization formats
                    agent_data = n.get("agent")

                    if isinstance(agent_data, dict):
                        if "serialization_error" in agent_data:
                            logger.warning(
                                f"Node {n['id']} was exported with agent serialization error: {agent_data['serialization_error']}"
                            )
                            # Create a placeholder agent or handle the error appropriately
                            agent = None  # Could create a dummy agent here
                        elif (
                            "agent_name" in agent_data
                            and "agent_type" in agent_data
                        ):
                            # This is a minimal agent representation
                            logger.info(
                                f"Node {n['id']} using simplified agent representation: {agent_data['agent_name']}"
                            )
                            agent = agent_data  # Store the dict representation for now
                        else:
                            # This should be a full agent dict
                            agent = agent_data
                    else:
                        # Legacy string representation
                        agent = agent_data

                    node = Node(
                        id=n["id"],
                        type=NodeType(n["type"]),
                        agent=agent,
                        metadata=n.get("metadata", {}),
                    )
                    nodes.append(node)

                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize node {n.get('id', 'unknown')}: {e}"
                    )
                    continue

            # Reconstruct edges
            edges = []
            for e in data["edges"]:
                try:
                    edge = Edge(
                        source=e["source"],
                        target=e["target"],
                        metadata=e.get("metadata", {}),
                    )
                    edges.append(edge)
                except Exception as ex:
                    logger.warning(
                        f"Failed to deserialize edge {e.get('source', 'unknown')} -> {e.get('target', 'unknown')}: {ex}"
                    )
                    continue

            # Extract all parameters with backward compatibility
            workflow_params = {
                "id": data.get("id"),
                "name": data.get("name", "Loaded-Workflow"),
                "description": data.get(
                    "description", "Workflow loaded from JSON"
                ),
                "entry_points": data.get("entry_points"),
                "end_points": data.get("end_points"),
                "max_loops": data.get("max_loops", 1),
                "task": data.get("task"),
                "auto_compile": data.get("auto_compile", True),
                "verbose": data.get("verbose", False),
            }

            # Create workflow using from_spec for proper initialization
            result = cls.from_spec(
                [n.agent for n in nodes if n.agent is not None],
                edges,
                **{
                    k: v
                    for k, v in workflow_params.items()
                    if v is not None
                },
            )

            # Restore additional parameters not handled by from_spec
            if "max_workers" in data:
                result._max_workers = data["max_workers"]
                if result.verbose:
                    logger.debug(
                        f"Restored max_workers: {result._max_workers}"
                    )

            # Restore conversation if present
            if "conversation" in data and data["conversation"]:
                try:
                    from swarms.structs.conversation import (
                        Conversation,
                    )

                    if isinstance(data["conversation"], dict):
                        if "history" in data["conversation"]:
                            # Reconstruct conversation from history
                            conv = Conversation()
                            conv.history = data["conversation"][
                                "history"
                            ]
                            result.conversation = conv
                            if result.verbose:
                                logger.debug(
                                    f"Restored conversation with {len(conv.history)} messages"
                                )
                        else:
                            logger.warning(
                                "Conversation data present but in unrecognized format"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to restore conversation: {e}"
                    )

            # Restore runtime state if requested
            if restore_runtime_state and "runtime_state" in data:
                runtime_state = data["runtime_state"]
                try:
                    if runtime_state.get("is_compiled", False):
                        result._compiled = True
                        result._compilation_timestamp = (
                            runtime_state.get("compilation_timestamp")
                        )
                        result._sorted_layers = runtime_state.get(
                            "sorted_layers", []
                        )

                        if result.verbose:
                            logger.info(
                                f"Restored runtime state: compiled={result._compiled}, layers={len(result._sorted_layers)}"
                            )
                    else:
                        if result.verbose:
                            logger.debug(
                                "Runtime state indicates workflow was not compiled"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to restore runtime state: {e}"
                    )

            # Log metrics if available
            if "metrics" in data:
                metrics = data["metrics"]
                logger.info(
                    f"Successfully loaded GraphWorkflow: {metrics.get('node_count', len(nodes))} nodes, "
                    f"{metrics.get('edge_count', len(edges))} edges, "
                    f"schema_version: {schema_version}"
                )
            else:
                logger.info(
                    f"Successfully loaded GraphWorkflow: {len(nodes)} nodes, {len(edges)} edges"
                )

            logger.success(
                "GraphWorkflow deserialization completed successfully"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON format in GraphWorkflow.from_json: {e}"
            )
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            logger.exception(f"Error in GraphWorkflow.from_json: {e}")
            raise e

    def get_compilation_status(self) -> Dict[str, Any]:
        """
        Get detailed compilation status information for debugging and monitoring.

        Returns:
            Dict[str, Any]: Compilation status including cache state, timestamps, and performance metrics.
        """
        status = {
            "is_compiled": self._compiled,
            "compilation_timestamp": self._compilation_timestamp,
            "cached_layers_count": (
                len(self._sorted_layers) if self._compiled else 0
            ),
            "max_workers": self._max_workers,
            "max_loops": self.max_loops,
            "cache_efficient": self._compiled and self.max_loops > 1,
        }

        if self._compilation_timestamp:
            status["time_since_compilation"] = (
                time.time() - self._compilation_timestamp
            )

        if self._compiled:
            status["layers"] = self._sorted_layers
            status["entry_points"] = self.entry_points
            status["end_points"] = self.end_points

        return status

    def save_to_file(
        self,
        filepath: str,
        include_conversation: bool = False,
        include_runtime_state: bool = False,
        overwrite: bool = False,
    ) -> str:
        """
        Save the workflow to a JSON file with comprehensive metadata.

        Args:
            filepath (str): Path to save the JSON file
            include_conversation (bool): Whether to include conversation history
            include_runtime_state (bool): Whether to include runtime compilation state
            overwrite (bool): Whether to overwrite existing files

        Returns:
            str: Path to the saved file

        Raises:
            FileExistsError: If file exists and overwrite is False
            Exception: If save operation fails
        """

        # Handle file path validation
        if not filepath.endswith(".json"):
            filepath += ".json"

        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"File {filepath} already exists. Set overwrite=True to replace it."
            )

        if self.verbose:
            logger.info(f"Saving GraphWorkflow to {filepath}")

        try:
            # Generate JSON with requested options
            json_data = self.to_json(
                fast=True,
                include_conversation=include_conversation,
                include_runtime_state=include_runtime_state,
            )

            # Create directory if it doesn't exist
            os.makedirs(
                os.path.dirname(os.path.abspath(filepath)),
                exist_ok=True,
            )

            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_data)

            file_size = os.path.getsize(filepath)
            logger.success(
                f"GraphWorkflow saved to {filepath} ({file_size:,} bytes)"
            )

            return filepath

        except Exception as e:
            logger.exception(
                f"Failed to save GraphWorkflow to {filepath}: {e}"
            )
            raise e

    @classmethod
    def load_from_file(
        cls, filepath: str, restore_runtime_state: bool = False
    ) -> "GraphWorkflow":
        """
        Load a workflow from a JSON file.

        Args:
            filepath (str): Path to the JSON file
            restore_runtime_state (bool): Whether to restore runtime compilation state

        Returns:
            GraphWorkflow: Loaded workflow instance

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If load operation fails
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Workflow file not found: {filepath}"
            )

        logger.info(f"Loading GraphWorkflow from {filepath}")

        try:
            # Read file
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = f.read()

            # Deserialize workflow
            workflow = cls.from_json(
                json_data, restore_runtime_state=restore_runtime_state
            )

            file_size = os.path.getsize(filepath)
            logger.success(
                f"GraphWorkflow loaded from {filepath} ({file_size:,} bytes)"
            )

            return workflow

        except Exception as e:
            logger.exception(
                f"Failed to load GraphWorkflow from {filepath}: {e}"
            )
            raise e

    def validate(self, auto_fix: bool = False) -> Dict[str, Any]:
        """
        Validate the workflow structure, checking for potential issues such as isolated nodes,
        cyclic dependencies, etc.

        Args:
            auto_fix (bool): Whether to automatically fix some simple issues (like auto-setting entry/exit points)

        Returns:
            Dict[str, Any]: Dictionary containing validation results, including validity, warnings and errors
        """
        if self.verbose:
            logger.debug(
                f"Validating GraphWorkflow structure (auto_fix={auto_fix})"
            )

        result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "fixed": [],
        }

        try:
            # Check for empty graph
            if not self.nodes:
                result["errors"].append("Workflow has no nodes")
                result["is_valid"] = False
                return result

            if not self.edges:
                result["warnings"].append(
                    "Workflow has no edges between nodes"
                )

            # Check for node agent instance validity
            invalid_agents = []
            for node_id, node in self.nodes.items():
                if node.agent is None:
                    invalid_agents.append(node_id)

            if invalid_agents:
                result["errors"].append(
                    f"Found {len(invalid_agents)} nodes with invalid agent instances: {invalid_agents}"
                )
                result["is_valid"] = False

            # Check for isolated nodes (no incoming or outgoing edges)
            isolated = [
                n
                for n in self.nodes
                if self.graph_backend.in_degree(n) == 0
                and self.graph_backend.out_degree(n) == 0
            ]
            if isolated:
                result["warnings"].append(
                    f"Found {len(isolated)} isolated nodes: {isolated}"
                )

            # Check for cyclic dependencies
            try:
                cycles = self.graph_backend.simple_cycles()
                if cycles:
                    result["warnings"].append(
                        f"Found {len(cycles)} cycles in workflow"
                    )
                    result["cycles"] = cycles
            except Exception as e:
                result["warnings"].append(
                    f"Could not check for cycles: {e}"
                )

            # Check entry points
            if not self.entry_points:
                result["warnings"].append("No entry points defined")
                if auto_fix:
                    self.auto_set_entry_points()
                    result["fixed"].append("Auto-set entry points")

            # Check exit points
            if not self.end_points:
                result["warnings"].append("No end points defined")
                if auto_fix:
                    self.auto_set_end_points()
                    result["fixed"].append("Auto-set end points")

            # Check for unreachable nodes (not reachable from entry points)
            if self.entry_points:
                reachable = set()
                for entry in self.entry_points:
                    reachable.update(
                        self.graph_backend.descendants(entry)
                    )
                    reachable.add(entry)

                unreachable = set(self.nodes.keys()) - reachable
                if unreachable:
                    result["warnings"].append(
                        f"Found {len(unreachable)} nodes unreachable from entry points: {unreachable}"
                    )
                    if auto_fix and unreachable:
                        # Add unreachable nodes as entry points
                        updated_entries = self.entry_points + list(
                            unreachable
                        )
                        self.set_entry_points(updated_entries)
                        result["fixed"].append(
                            f"Added {len(unreachable)} unreachable nodes to entry points"
                        )

            # Check for dead-end nodes (cannot reach any exit point)
            if self.end_points:
                reverse_graph = self.graph_backend.reverse()
                reachable_to_exit = set()
                for exit_point in self.end_points:
                    reachable_to_exit.update(
                        reverse_graph.descendants(exit_point)
                    )
                    reachable_to_exit.add(exit_point)

                dead_ends = set(self.nodes.keys()) - reachable_to_exit
                if dead_ends:
                    result["warnings"].append(
                        f"Found {len(dead_ends)} nodes that cannot reach any exit point: {dead_ends}"
                    )
                    if auto_fix and dead_ends:
                        # Add dead-end nodes as exit points
                        updated_exits = self.end_points + list(
                            dead_ends
                        )
                        self.set_end_points(updated_exits)
                        result["fixed"].append(
                            f"Added {len(dead_ends)} dead-end nodes to exit points"
                        )

            # Check for serious warnings
            has_serious_warnings = any(
                "cycle" in warning.lower()
                or "unreachable" in warning.lower()
                for warning in result["warnings"]
            )

            # If there are errors or serious warnings without fixes, the workflow is invalid
            if result["errors"] or (
                has_serious_warnings and not auto_fix
            ):
                result["is_valid"] = False

            if self.verbose:
                if result["is_valid"]:
                    if result["warnings"]:
                        logger.warning(
                            f"Validation found {len(result['warnings'])} warnings but workflow is still valid"
                        )
                    else:
                        logger.success(
                            "Workflow validation completed with no issues"
                        )
                else:
                    logger.error(
                        f"Validation found workflow to be invalid with {len(result['errors'])} errors and {len(result['warnings'])} warnings"
                    )

                if result["fixed"]:
                    logger.info(
                        f"Auto-fixed {len(result['fixed'])} issues: {', '.join(result['fixed'])}"
                    )

            return result
        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(str(e))
            logger.exception(f"Error during workflow validation: {e}")
            return result

    def export_summary(self) -> Dict[str, Any]:
        """
        Generate a human-readable summary of the workflow for inspection.

        Returns:
            Dict[str, Any]: Comprehensive workflow summary
        """
        summary = {
            "workflow_info": {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "created": getattr(self, "_creation_time", "unknown"),
            },
            "structure": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "entry_points": len(self.entry_points),
                "end_points": len(self.end_points),
                "layers": (
                    len(self._sorted_layers)
                    if self._compiled
                    else "not compiled"
                ),
            },
            "configuration": {
                "max_loops": self.max_loops,
                "max_workers": self._max_workers,
                "auto_compile": self.auto_compile,
                "verbose": self.verbose,
            },
            "compilation_status": self.get_compilation_status(),
            "agents": [
                {
                    "id": node.id,
                    "type": str(node.type),
                    "agent_name": getattr(
                        node.agent, "agent_name", "unknown"
                    ),
                    "agent_type": str(type(node.agent)),
                }
                for node in self.nodes.values()
            ],
            "connections": [
                {
                    "from": edge.source,
                    "to": edge.target,
                    "metadata": edge.metadata,
                }
                for edge in self.edges
            ],
        }

        # Add task info if available
        if self.task:
            summary["task"] = {
                "defined": True,
                "length": len(str(self.task)),
                "preview": (
                    str(self.task)[:100] + "..."
                    if len(str(self.task)) > 100
                    else str(self.task)
                ),
            }
        else:
            summary["task"] = {"defined": False}

        # Add conversation info if available
        if self.conversation:
            try:
                if hasattr(self.conversation, "history"):
                    summary["conversation"] = {
                        "available": True,
                        "message_count": len(
                            self.conversation.history
                        ),
                        "type": str(type(self.conversation)),
                    }
                else:
                    summary["conversation"] = {
                        "available": True,
                        "message_count": "unknown",
                        "type": str(type(self.conversation)),
                    }
            except Exception as e:
                summary["conversation"] = {
                    "available": True,
                    "error": str(e),
                }
        else:
            summary["conversation"] = {"available": False}

        return summary
