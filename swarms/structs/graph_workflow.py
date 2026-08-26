import asyncio
import concurrent.futures
import hashlib
import json
import os
import time
import traceback
import uuid
from pathlib import Path
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
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
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_init,
    trace_run,
)
from swarms.utils.generate_id import generate_id

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

    def is_dag(self) -> bool:
        """
        Report whether the graph is acyclic.

        Cheap O(V+E) alternative to enumerating every simple cycle, which is
        exponential in the number of cycles.

        Returns:
            bool: True when the graph contains no directed cycle.
        """
        raise NotImplementedError

    def adjacency(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Materialise successor and predecessor maps in a single pass.

        Callers that need repeated neighbour lookups (validation, prompt
        building) use this instead of paying per-node backend calls.

        Returns:
            Tuple[Dict[str, List[str]], Dict[str, List[str]]]: ``(successors,
            predecessors)`` keyed by node ID. Every node is present, including
            isolated ones.
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
        return self.graph.in_degree(node_id)

    def out_degree(self, node_id: str) -> int:
        return self.graph.out_degree(node_id)

    def predecessors(self, node_id: str) -> Iterator[str]:
        return self.graph.predecessors(node_id)

    def reverse(self) -> "NetworkXBackend":
        """
        Return a reversed copy of the graph.

        Returns:
            NetworkXBackend: A new backend instance with reversed edges.
        """
        reversed_backend = NetworkXBackend()
        # copy=False avoids deepcopy of node attributes, which fails for
        # Agent objects that hold unpicklable thread locks.
        reversed_backend.graph = self.graph.reverse(copy=False)
        return reversed_backend

    def topological_generations(self) -> List[List[str]]:
        """
        Get topological generations (layers) of the graph.

        Returns:
            List[List[str]]: List of layers, where each layer is a list of node
            IDs. A cyclic remainder, if any, is appended as a final layer
            rather than raising.
        """
        try:
            return list(nx.topological_generations(self.graph))
        except nx.NetworkXUnfeasible:
            # Cyclic graph: layer what we can with Kahn's algorithm and append
            # the cyclic remainder so callers still get an execution order.
            graph = self.graph
            indegree = dict(graph.in_degree())
            frontier = [n for n, d in indegree.items() if d == 0]
            layers: List[List[str]] = []
            placed = 0
            while frontier:
                layers.append(frontier)
                placed += len(frontier)
                nxt = []
                for node_id in frontier:
                    for succ_id in graph.successors(node_id):
                        indegree[succ_id] -= 1
                        if indegree[succ_id] == 0:
                            nxt.append(succ_id)
                frontier = nxt
            if placed < graph.number_of_nodes():
                layers.append(
                    [n for n, d in indegree.items() if d > 0]
                )
            return layers

    def simple_cycles(self) -> List[List[str]]:
        # Enumerating simple cycles is exponential in the number of cycles, so
        # short-circuit on the overwhelmingly common acyclic case first.
        if nx.is_directed_acyclic_graph(self.graph):
            return []
        return list(nx.simple_cycles(self.graph))

    def descendants(self, node_id: str) -> Set[str]:
        return nx.descendants(self.graph, node_id)

    def is_dag(self) -> bool:
        """
        Report whether the graph is acyclic.

        Returns:
            bool: True when the graph contains no directed cycle.
        """
        return nx.is_directed_acyclic_graph(self.graph)

    def adjacency(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Materialise successor and predecessor maps in a single pass.

        Returns:
            Tuple[Dict[str, List[str]], Dict[str, List[str]]]: ``(successors,
            predecessors)`` keyed by node ID.
        """
        graph = self.graph
        succ: Dict[str, List[str]] = {n: [] for n in graph}
        pred: Dict[str, List[str]] = {n: [] for n in graph}
        for source, target in graph.edges():
            succ[source].append(target)
            pred[target].append(source)
        return succ, pred


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
        if node_id not in self._node_id_to_index:
            return 0
        node_index = self._node_id_to_index[node_id]
        return self.graph.in_degree(node_index)

    def out_degree(self, node_id: str) -> int:
        if node_id not in self._node_id_to_index:
            return 0
        node_index = self._node_id_to_index[node_id]
        return self.graph.out_degree(node_index)

    def predecessors(self, node_id: str) -> Iterator[str]:
        if node_id not in self._node_id_to_index:
            return iter([])
        target_index = self._node_id_to_index[node_id]
        index_to_id = self._index_to_node_id
        # predecessor_indices is O(in-degree); scanning the full edge list
        # would make this O(E) per node.
        return iter(
            [
                index_to_id[idx]
                for idx in self.graph.predecessor_indices(
                    target_index
                )
            ]
        )

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
        all_indices = list(self._node_id_to_index.values())
        if not all_indices:
            return []

        index_to_id = self._index_to_node_id

        try:
            # rustworkx computes generations natively in Rust — O(V+E).
            layers = [
                [index_to_id[idx] for idx in generation]
                for generation in rx.topological_generations(
                    self.graph
                )
            ]
        except Exception as e:
            # Raised when the graph contains a cycle. Fall back to Kahn's
            # algorithm so the cyclic remainder is still surfaced as a final
            # layer, matching the previous behaviour.
            logger.warning(
                f"rustworkx topological_generations failed ({e}); "
                "falling back to Kahn's algorithm"
            )
            return self._kahn_generations(all_indices)

        placed = sum(len(layer) for layer in layers)
        if placed < len(all_indices):
            # Cyclic remainder: append the unplaced nodes as a final layer.
            seen = {nid for layer in layers for nid in layer}
            layers.append(
                [
                    index_to_id[idx]
                    for idx in all_indices
                    if index_to_id[idx] not in seen
                ]
            )

        return layers or [[index_to_id[idx] for idx in all_indices]]

    def _kahn_generations(
        self, all_indices: List[int]
    ) -> List[List[str]]:
        """
        Layer the graph with Kahn's algorithm, tolerating cycles.

        Args:
            all_indices (List[int]): Every node index in the graph.

        Returns:
            List[List[str]]: Layers of node IDs; any cyclic remainder is
            appended as a final layer.
        """
        index_to_id = self._index_to_node_id
        graph = self.graph

        indegree = {idx: graph.in_degree(idx) for idx in all_indices}
        frontier = [idx for idx in all_indices if indegree[idx] == 0]
        layers: List[List[str]] = []
        placed = 0

        while frontier:
            layers.append([index_to_id[idx] for idx in frontier])
            placed += len(frontier)
            nxt = []
            for idx in frontier:
                for succ_idx in graph.successor_indices(idx):
                    indegree[succ_idx] -= 1
                    if indegree[succ_idx] == 0:
                        nxt.append(succ_idx)
            frontier = nxt

        if placed < len(all_indices):
            layers.append(
                [
                    index_to_id[idx]
                    for idx in all_indices
                    if indegree[idx] > 0
                ]
            )

        return layers or [[index_to_id[idx] for idx in all_indices]]

    def simple_cycles(self) -> List[List[str]]:
        try:
            # Enumerating cycles is exponential in their count; the acyclic
            # check is O(V+E) and covers the common case.
            if rx.is_directed_acyclic_graph(self.graph):
                return []
            index_to_id = self._index_to_node_id
            return [
                [index_to_id[idx] for idx in cycle]
                for cycle in rx.simple_cycles(self.graph)
            ]
        except Exception as e:
            logger.warning(
                f"Error in rustworkx simple_cycles: {e}, returning empty list"
            )
            return []

    def descendants(self, node_id: str) -> Set[str]:
        if node_id not in self._node_id_to_index:
            return set()
        node_index = self._node_id_to_index[node_id]
        index_to_id = self._index_to_node_id
        # Native Rust traversal; the previous hand-rolled BFS also used
        # list.pop(0), which is O(n) per dequeue.
        return {
            index_to_id[idx]
            for idx in rx.descendants(self.graph, node_index)
        }

    def is_dag(self) -> bool:
        """
        Report whether the graph is acyclic.

        Returns:
            bool: True when the graph contains no directed cycle.
        """
        return rx.is_directed_acyclic_graph(self.graph)

    def adjacency(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Materialise successor and predecessor maps in a single pass.

        Returns:
            Tuple[Dict[str, List[str]], Dict[str, List[str]]]: ``(successors,
            predecessors)`` keyed by node ID.
        """
        index_to_id = self._index_to_node_id
        succ: Dict[str, List[str]] = {
            nid: [] for nid in self._node_id_to_index
        }
        pred: Dict[str, List[str]] = {
            nid: [] for nid in self._node_id_to_index
        }
        for source_idx, target_idx in self.graph.edge_list():
            source_id = index_to_id[source_idx]
            target_id = index_to_id[target_idx]
            succ[source_id].append(target_id)
            pred[target_id].append(source_id)
        return succ, pred


class NodeType(str, Enum):
    AGENT = "agent"
    SUBGRAPH = "subgraph"


class Node:
    """
    Represents a node in a graph workflow.  A node can be either an Agent or
    a nested GraphWorkflow (subgraph).

    Attributes:
        id (str): The unique identifier of the node.
        type (NodeType): AGENT or SUBGRAPH.
        agent (Any): The Agent or GraphWorkflow associated with the node.
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
            agent (Any, optional): The Agent or GraphWorkflow associated with the node.
            metadata (Dict[str, Any], optional): Additional metadata for the node.
        """
        self.id = id
        self.type = type
        self.agent = agent
        self.metadata = metadata or {}

        if not self.id:
            if self.agent is not None:
                self.id = getattr(
                    self.agent, "agent_name", None
                ) or getattr(self.agent, "name", None)
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

    @classmethod
    def from_subgraph(
        cls,
        subgraph: "GraphWorkflow",
        node_id: str = None,
        **kwargs: Any,
    ) -> "Node":
        """
        Create a Node from a nested GraphWorkflow.

        Args:
            subgraph (GraphWorkflow): The inner workflow to embed as a node.
            node_id (str, optional): Override ID; defaults to subgraph.name.
            **kwargs: Additional keyword arguments.

        Returns:
            Node: A new SUBGRAPH Node instance.
        """
        return cls(
            type=NodeType.SUBGRAPH,
            agent=subgraph,
            id=node_id or getattr(subgraph, "name", None),
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
        _max_workers (int): Maximum nodes executed concurrently. Set from max_parallel_nodes if provided, otherwise int(get_cpu_cores() * 0.95) with a minimum of 1.
        verbose (bool): Whether to enable verbose logging.
    """

    def __init__(
        self,
        id: Optional[str] = None,
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
        checkpoint_dir: Optional[str] = None,
        on_node_complete: Optional[Callable[[str, Any], None]] = None,
        max_parallel_nodes: Optional[int] = None,
    ):
        self.id = id or generate_id("graph-workflow")
        self.verbose = verbose
        self.on_node_complete = on_node_complete

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

        # Checkpoint configuration
        self.checkpoint_dir = checkpoint_dir

        # Private optimization attributes
        self._compiled = False
        self._sorted_layers = []
        self._execution_plan = []
        self._successors_map = {}
        self._predecessors_cache = {}
        self.max_parallel_nodes = max_parallel_nodes
        self._max_workers = (
            max(1, max_parallel_nodes)
            if max_parallel_nodes is not None
            else max(1, int(get_cpu_cores() * 0.95))
        )
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

        # Capture the full __init__ configuration if telemetry is enabled.
        capture_init(self)

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
        self._execution_plan = []
        self._successors_map = {}
        self._compilation_timestamp = None

        # Clear predecessors cache when graph structure changes
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

            # Build the adjacency maps once and reuse them for validation and
            # for per-node predecessor lookups during execution.
            succ, pred = self.graph_backend.adjacency()
            self._successors_map = succ
            self._predecessors_cache = {
                node_id: tuple(parents)
                for node_id, parents in pred.items()
            }

            # Structural validation, surfaced at build time rather than
            # mid-execution.  We never raise here so that compile() stays
            # backward-compatible; callers that want strict enforcement
            # should call validate(raise_on_error=True) explicitly.
            if self.nodes:
                errors, warnings = self._fast_validate(succ, pred)
                if errors:
                    logger.warning(
                        f"GraphWorkflow compile: validation found "
                        f"{len(errors)} error(s) — "
                        + "; ".join(errors)
                    )
                if warnings and self.verbose:
                    logger.warning(
                        f"GraphWorkflow compile: validation found "
                        f"{len(warnings)} warning(s) — "
                        + "; ".join(warnings)
                    )

            # Freeze the per-layer execution plan so run() does no dictionary
            # lookups or attribute resolution in its hot loop.
            self._execution_plan = [
                [
                    (
                        node_id,
                        self.nodes[node_id].agent,
                        self.nodes[node_id].type,
                        getattr(
                            self.nodes[node_id].agent,
                            "agent_name",
                            node_id,
                        ),
                    )
                    for node_id in layer
                    if node_id in self.nodes
                ]
                for layer in sorted_layers
            ]

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

    @staticmethod
    def _multi_source_reachable(
        sources: List[str], adjacency: Dict[str, List[str]]
    ) -> Set[str]:
        """
        Nodes reachable from any of ``sources`` in one traversal.

        A single multi-source BFS replaces one traversal per source, which is
        what the per-entry-point/per-end-point loops used to cost.

        Args:
            sources (List[str]): Starting node IDs.
            adjacency (Dict[str, List[str]]): Neighbour map to walk.

        Returns:
            Set[str]: All reachable node IDs, including the sources themselves.
        """
        seen = set(sources)
        frontier = list(sources)
        while frontier:
            node_id = frontier.pop()
            for neighbour in adjacency.get(node_id, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
        return seen

    def _structural_checks(
        self,
        succ: Dict[str, List[str]],
        pred: Dict[str, List[str]],
        *,
        enumerate_cycles: bool = False,
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        The checks shared by :meth:`validate` and :meth:`_fast_validate`, so the
        compile-time and user-facing checks cannot drift apart.

        Args:
            succ: Successor map from ``adjacency()``.
            pred: Predecessor map from ``adjacency()``.
            enumerate_cycles: List every simple cycle instead of the O(V+E)
                "is there one at all" test. Exponential in the cycle count,
                so :meth:`compile` leaves it off.

        Returns:
            ``(errors, warnings, details)``. ``details`` holds the offending
            node sets so callers need not re-derive them from the strings.
        """
        errors: List[str] = []
        warnings: List[str] = []
        cycles: List[List[str]] = []

        invalid_agents = [
            node_id
            for node_id, node in self.nodes.items()
            if node.agent is None
        ]
        if invalid_agents:
            errors.append(
                f"Found {len(invalid_agents)} nodes with invalid agent "
                f"instances: {invalid_agents}"
            )

        if not self.edges:
            warnings.append("Workflow has no edges between nodes")

        isolated = [
            node_id
            for node_id in self.nodes
            if not succ.get(node_id) and not pred.get(node_id)
        ]
        if isolated:
            warnings.append(
                f"Found {len(isolated)} isolated nodes: {isolated}"
            )

        try:
            if enumerate_cycles:
                cycles = self.graph_backend.simple_cycles()
                if cycles:
                    warnings.append(
                        f"Found {len(cycles)} cycles in workflow"
                    )
            elif not self.graph_backend.is_dag():
                warnings.append("Found cycles in workflow")
        except Exception as e:
            warnings.append(f"Could not check for cycles: {e}")

        all_ids = set(self.nodes.keys())

        unreachable: List[str] = []
        if self.entry_points:
            unreachable = sorted(
                all_ids
                - self._multi_source_reachable(
                    self.entry_points, succ
                )
            )
            if unreachable:
                warnings.append(
                    f"Found {len(unreachable)} nodes unreachable from "
                    f"entry points: {set(unreachable)}"
                )

        dead_ends: List[str] = []
        if self.end_points:
            dead_ends = sorted(
                all_ids
                - self._multi_source_reachable(self.end_points, pred)
            )
            if dead_ends:
                warnings.append(
                    f"Found {len(dead_ends)} nodes that cannot reach any "
                    f"exit point: {set(dead_ends)}"
                )

        return (
            errors,
            warnings,
            {
                "cycles": cycles,
                "isolated": isolated,
                "unreachable": unreachable,
                "dead_ends": dead_ends,
            },
        )

    def _fast_validate(
        self,
        succ: Dict[str, List[str]],
        pred: Dict[str, List[str]],
    ) -> Tuple[List[str], List[str]]:
        """Same coverage as :meth:`validate`, off the maps :meth:`compile`
        already built — no backend calls, no reversed copy, no cycle
        enumeration. Returns ``(errors, warnings)``."""
        errors, warnings, _ = self._structural_checks(succ, pred)
        return errors, warnings

    def add_node(
        self,
        agent: Union[Agent, "GraphWorkflow"],
        **kwargs: Any,
    ) -> None:
        """
        Adds an agent or a nested GraphWorkflow as a node in the workflow graph.

        Args:
            agent (Union[Agent, GraphWorkflow]): The agent or inner workflow to add.
            **kwargs: Additional keyword arguments for the node.
        """
        if self.verbose:
            label = getattr(agent, "agent_name", None) or getattr(
                agent, "name", "unnamed"
            )
            logger.debug(f"Adding node: {label}")

        try:
            if isinstance(agent, GraphWorkflow):
                node = Node.from_subgraph(agent, **kwargs)
                if not agent._compiled:
                    agent.compile()
            else:
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
                f"Error in GraphWorkflow.add_node for {getattr(agent, 'agent_name', getattr(agent, 'name', 'unnamed'))}: {e}"
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
            with ContextThreadPoolExecutor(
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

    def _attach_edge(self, edge: Edge) -> Edge:
        """
        Register one edge, after checking both endpoints exist.

        The four public edge builders differ only in which pairs they generate;
        this is the part they all shared verbatim.

        Raises:
            ValueError: If either endpoint is not a node in this workflow.
        """
        for role, node_id in (
            ("Source", edge.source),
            ("Target", edge.target),
        ):
            if node_id not in self.nodes:
                error_msg = f"{role} node '{node_id}' does not exist in GraphWorkflow"
                logger.error(error_msg)
                raise ValueError(error_msg)

        self.edges.append(edge)
        self.graph_backend.add_edge(
            edge.source, edge.target, **(edge.metadata or {})
        )
        return edge

    def _add_edge_pairs(
        self,
        pairs: Iterable[Tuple[Any, Any]],
        kind: str,
        **kwargs: Any,
    ) -> List[Edge]:
        """Attach one edge per ``(source, target)`` pair, invalidating the
        compilation cache once at the end rather than per edge."""
        created = [
            self._attach_edge(
                Edge.from_nodes(source, target, **kwargs)
            )
            for source, target in pairs
        ]
        self._invalidate_compilation()
        if self.verbose:
            logger.success(f"Added {len(created)} {kind} edges")
        return created

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

            self._attach_edge(edge)
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
        return self._add_edge_pairs(
            ((source, target) for target in targets),
            "fan-out",
            **kwargs,
        )

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
        return self._add_edge_pairs(
            ((source, target) for source in sources),
            "fan-in",
            **kwargs,
        )

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
        return self._add_edge_pairs(
            (
                (source, target)
                for source in sources
                for target in targets
            ),
            "parallel-chain",
            **kwargs,
        )

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
        # Instance-level caching instead of @lru_cache to avoid hashing issues.
        # compile() populates this map wholesale from a single adjacency pass;
        # this path only fills gaps for nodes queried before compilation.
        cache = self._predecessors_cache
        preds = cache.get(node_id)
        if preds is None:
            preds = tuple(self.graph_backend.predecessors(node_id))
            cache[node_id] = preds
        return preds

    def _build_prompt(
        self,
        node_id: str,
        task: str,
        prev_outputs: Dict[str, Any],
        layer_idx: int,
        loop_idx: int = 0,
    ) -> str:
        """
        Optimized prompt building with minimal string operations.

        Args:
            node_id (str): The node ID to build a prompt for.
            task (str): The main task.
            prev_outputs (Dict[str, Any]): Previous outputs from predecessor nodes.
                For loop_idx > 0 this also contains end-point outputs from the
                previous loop iteration.
            layer_idx (int): The current layer index.
            loop_idx (int): The current loop iteration (0-based).

        Returns:
            str: The built prompt.
        """
        if self.verbose:
            logger.debug(
                f"Building prompt for node {node_id} (layer {layer_idx}, loop {loop_idx})"
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
            elif loop_idx > 0 and layer_idx == 0 and prev_outputs:
                # Entry-point nodes in subsequent loops receive end-point
                # outputs from the previous loop as refinement context.
                prior_parts = [
                    f"Output from {nid} (previous iteration):\n{out}"
                    for nid, out in prev_outputs.items()
                    if out is not None
                ]
                prior_context = "\n\n".join(prior_parts)

                prompt = (
                    f"Original Task: {task}\n\n"
                    f"Previous Iteration Outputs:\n{prior_context}\n\n"
                    f"Instructions: This is iteration {loop_idx + 1} of the workflow. "
                    f"Review the outputs from the previous iteration above. "
                    f"Refine, correct, or expand upon the previous results. "
                    f"Focus on improving accuracy, filling gaps, and strengthening the analysis."
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
            result = await asyncio.to_thread(
                self.run, task, *args, **kwargs
            )

            if self.verbose:
                logger.success(
                    "Async GraphWorkflow execution completed"
                )

            return result

        except Exception as e:
            logger.exception(f"Error in GraphWorkflow.arun: {e}")
            raise e

    @staticmethod
    def _task_key(task: str) -> str:
        """Stable per-task checkpoint key. Python's ``hash()`` is salted and
        would not survive a process restart."""
        return hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]

    def _checkpoint_path(self, task_key: str, layer_idx: int) -> Path:
        """Where one layer's checkpoint lives."""
        return (
            Path(self.checkpoint_dir)
            / f"{task_key}_layer_{layer_idx}.json"
        )

    @staticmethod
    def _safe_output(
        agent_name: str, produce: Callable[[], Any]
    ) -> Tuple[bool, Any]:
        """
        Run ``produce``, turning a failure into an ``[ERROR]`` string.

        One node blowing up degrades that node's output rather than the whole
        layer, so downstream agents still receive something to work with.
        Returns ``(succeeded, output)``.
        """
        try:
            return True, produce()
        except Exception as e:
            logger.exception(
                f"Error in GraphWorkflow agent execution for {agent_name}: {e}"
            )
            return False, f"[ERROR] Agent {agent_name} failed: {e}"

    @trace_run(
        "GraphWorkflow.run",
        input_params=("task", "tasks", "img", "imgs"),
    )
    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        on_node_complete: Optional[Callable[[str, Any], None]] = None,
        streaming_callback: Optional[
            Callable[[str, str], None]
        ] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run the workflow graph with optimized parallel agent execution.

        When max_loops > 1, the graph is executed multiple times. End-point
        outputs from each loop are fed as additional context into the next
        loop so that agents can iteratively refine their results.

        Args:
            task (Optional[str]): Task to execute. Uses self.task if not provided.
            img (Optional[str]): Optional image path for multimodal tasks.
            on_node_complete (Optional[Callable[[str, Any], None]]): Callback
                fired immediately when each agent finishes, before the layer
                completes. Receives ``(node_id, output)``. A callback passed
                here takes precedence over the instance-level callback set in
                ``__init__``.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback
                fired for every token as agents generate output in real-time.
                Receives ``(node_id, token)``.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Execution results keyed by node ID.
                When max_loops == 1, returns the single loop's results.
                When max_loops > 1, returns a dict with per-loop results
                keyed as ``{node_id}_loop_{loop_number}`` plus the final
                loop's results under the plain ``node_id`` keys.
        """
        # Resolve callbacks: run-level overrides instance-level
        _on_node_complete = on_node_complete or self.on_node_complete
        _streaming_callback = streaming_callback
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

        # One executor for the whole run instead of one per layer per loop.
        # Thread creation and pool shutdown dominated execution time on graphs
        # with many layers.  Sized to the widest layer, capped at _max_workers,
        # and created lazily so purely sequential graphs spawn no threads.
        widest_layer = max(
            (len(layer) for layer in self._execution_plan), default=1
        )
        executor: Optional[ContextThreadPoolExecutor] = None

        def _get_executor() -> ContextThreadPoolExecutor:
            nonlocal executor
            if executor is None:
                executor = ContextThreadPoolExecutor(
                    max_workers=max(
                        1, min(self._max_workers, widest_layer)
                    ),
                    thread_name_prefix=f"graph-{self.name}",
                )
                if self.verbose:
                    logger.debug(
                        f"Created shared thread pool with "
                        f"{max(1, min(self._max_workers, widest_layer))} workers"
                    )
            return executor

        try:
            loop = 0
            # Accumulated results across all loops
            all_loop_results: Dict[str, Any] = {}
            # End-point outputs carried forward as context for the next loop
            prior_loop_end_outputs: Dict[str, Any] = {}

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

                # Derive a deterministic key for this task so checkpoints
                # survive process restarts (Python's hash() is salted and
                # is NOT stable across runs).  Only needed when checkpointing.
                task_key = (
                    self._task_key(task)
                    if self.checkpoint_dir
                    else None
                )

                # Seed entry-point nodes with end-point outputs from the
                # previous loop so agents can refine iteratively.
                if prior_loop_end_outputs:
                    prev_outputs.update(prior_loop_end_outputs)

                for layer_idx, layer in enumerate(
                    self._execution_plan
                ):
                    layer_start_time = time.time()

                    # ----------------------------------------------------------
                    # Checkpoint resume: if this layer already has a saved result
                    # for the current task, load it and skip re-execution.
                    # ----------------------------------------------------------
                    if self.checkpoint_dir:
                        checkpoint_path = self._checkpoint_path(
                            task_key, layer_idx
                        )
                        if checkpoint_path.exists():
                            try:
                                saved = json.loads(
                                    checkpoint_path.read_text(
                                        encoding="utf-8"
                                    )
                                )
                                prev_outputs.update(saved)
                                execution_results.update(saved)
                                # Replay into conversation so workflow state
                                # is identical to a non-checkpoint run.
                                for node_id, output in saved.items():
                                    agent_name = (
                                        getattr(
                                            self.nodes[node_id].agent,
                                            "agent_name",
                                            node_id,
                                        )
                                        if node_id in self.nodes
                                        else node_id
                                    )
                                    try:
                                        self.conversation.add(
                                            role=agent_name,
                                            content=output,
                                        )
                                    except Exception:
                                        pass
                                logger.info(
                                    f"Checkpoint found - skipping layer {layer_idx + 1} "
                                    f"({len(saved)} agents restored from {checkpoint_path})"
                                )
                                continue
                            except Exception as cp_err:
                                logger.warning(
                                    f"Failed to load checkpoint {checkpoint_path}, "
                                    f"re-executing layer: {cp_err}"
                                )

                    if self.verbose:
                        logger.info(
                            f"Executing layer {layer_idx + 1}/{len(self._execution_plan)} "
                            f"with {len(layer)} nodes: {[n[0] for n in layer]}"
                        )

                    # Pre-build all prompts for this layer
                    layer_data = []
                    for (
                        node_id,
                        agent,
                        node_type,
                        agent_name,
                    ) in layer:
                        try:
                            prompt = self._build_prompt(
                                node_id,
                                task,
                                prev_outputs,
                                layer_idx,
                                loop,
                            )
                        except Exception as e:
                            logger.exception(
                                f"Error building prompt for node {node_id}: {e}"
                            )
                            # Continue with an error prompt as fallback
                            prompt = f"Error building prompt: {e}"
                        layer_data.append(
                            (
                                node_id,
                                agent,
                                node_type,
                                agent_name,
                                prompt,
                            )
                        )

                    def _make_call(node_id, agent, node_type, prompt):
                        """Bind one node's invocation into a zero-arg callable."""
                        if node_type == NodeType.SUBGRAPH:
                            # Subgraphs receive the prompt as their task and
                            # run in isolation.  Checkpoint state is stored
                            # under a sub-directory keyed by the parent node.
                            inner: GraphWorkflow = agent
                            _prev_cp = inner.checkpoint_dir
                            if (
                                self.checkpoint_dir
                                and not inner.checkpoint_dir
                            ):
                                inner.checkpoint_dir = str(
                                    Path(self.checkpoint_dir)
                                    / node_id
                                )

                            def _run_inner(
                                _inner=inner,
                                _prompt=prompt,
                                _prev=_prev_cp,
                            ):
                                try:
                                    return _inner.run(
                                        _prompt,
                                        img=img,
                                        *args,
                                        **kwargs,
                                    )
                                finally:
                                    _inner.checkpoint_dir = _prev

                            return _run_inner

                        if _streaming_callback is None:
                            # Common path: no per-node kwargs copy needed.
                            def _run_agent(
                                _agent=agent, _prompt=prompt
                            ):
                                return _agent.run(
                                    _prompt, img, *args, **kwargs
                                )

                            return _run_agent

                        def _run_agent_streaming(
                            _agent=agent,
                            _prompt=prompt,
                            _nid=node_id,
                        ):
                            call_kwargs = dict(kwargs)
                            call_kwargs["streaming_callback"] = (
                                lambda token: _streaming_callback(
                                    _nid, token
                                )
                            )
                            return _agent.run(
                                _prompt, img, *args, **call_kwargs
                            )

                        return _run_agent_streaming

                    def _record(
                        node_id, agent_name, node_type, output
                    ):
                        """Persist one node's output into the run's state."""
                        # Subgraph nodes return a dict; flatten to a readable
                        # string for downstream agents.  Only applied to
                        # SUBGRAPH nodes so that agent nodes returning
                        # structured dicts are not silently coerced.
                        if (
                            node_type == NodeType.SUBGRAPH
                            and isinstance(output, dict)
                        ):
                            output = "\n\n".join(
                                f"[{k}]: {v}"
                                for k, v in output.items()
                                if v is not None
                            )

                        prev_outputs[node_id] = output
                        execution_results[node_id] = output

                        try:
                            self.conversation.add(
                                role=agent_name, content=output
                            )
                        except Exception as e:
                            logger.exception(
                                f"Error adding output to conversation for agent {agent_name}: {e}"
                            )

                        if _on_node_complete is not None:
                            try:
                                _on_node_complete(node_id, output)
                            except Exception as e:
                                logger.exception(
                                    f"Error in on_node_complete callback for {agent_name}: {e}"
                                )

                    if len(layer_data) == 1:
                        # Single-node layer: run inline. Dispatching to a
                        # worker thread would only add handoff latency.
                        (
                            node_id,
                            agent,
                            node_type,
                            agent_name,
                            prompt,
                        ) = layer_data[0]
                        _, output = self._safe_output(
                            agent_name,
                            _make_call(
                                node_id, agent, node_type, prompt
                            ),
                        )
                        _record(
                            node_id, agent_name, node_type, output
                        )
                    else:
                        # Parallel layer: dispatch onto the run-wide pool.
                        pool = _get_executor()
                        future_to_data = {}

                        for (
                            node_id,
                            agent,
                            node_type,
                            agent_name,
                            prompt,
                        ) in layer_data:
                            try:
                                future = pool.submit(
                                    _make_call(
                                        node_id,
                                        agent,
                                        node_type,
                                        prompt,
                                    )
                                )
                                future_to_data[future] = (
                                    node_id,
                                    agent_name,
                                    node_type,
                                )
                            except Exception as e:
                                logger.exception(
                                    f"Error submitting task for agent {agent_name}: {e}"
                                )
                                error_output = f"[ERROR] Failed to submit task: {e}"
                                prev_outputs[node_id] = error_output
                                execution_results[node_id] = (
                                    error_output
                                )

                        completed_count = 0
                        for future in concurrent.futures.as_completed(
                            future_to_data
                        ):
                            (
                                node_id,
                                agent_name,
                                node_type,
                            ) = future_to_data[future]
                            ok, output = self._safe_output(
                                agent_name, future.result
                            )
                            if ok:
                                completed_count += 1
                                if self.verbose:
                                    logger.success(
                                        f"Agent {agent_name} completed successfully "
                                        f"({completed_count}/{len(layer_data)})"
                                    )

                            _record(
                                node_id, agent_name, node_type, output
                            )

                    layer_execution_time = (
                        time.time() - layer_start_time
                    )

                    # ----------------------------------------------------------
                    # Checkpoint save: persist this layer's outputs so a crash
                    # on a later layer doesn't force re-running this one.
                    # ----------------------------------------------------------
                    if self.checkpoint_dir:
                        try:
                            Path(self.checkpoint_dir).mkdir(
                                parents=True, exist_ok=True
                            )
                            checkpoint_path = self._checkpoint_path(
                                task_key, layer_idx
                            )
                            layer_outputs = {
                                entry[0]: prev_outputs[entry[0]]
                                for entry in layer
                                if entry[0] in prev_outputs
                            }
                            checkpoint_path.write_text(
                                json.dumps(
                                    layer_outputs,
                                    indent=2,
                                    default=str,
                                ),
                                encoding="utf-8",
                            )
                            if self.verbose:
                                logger.info(
                                    f"Checkpoint saved for layer {layer_idx + 1} → {checkpoint_path}"
                                )
                        except Exception as cp_err:
                            logger.warning(
                                f"Failed to save checkpoint for layer {layer_idx + 1}: {cp_err}"
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

                # Capture end-point outputs to pass as context into the next loop
                prior_loop_end_outputs = {
                    node_id: execution_results[node_id]
                    for node_id in self.end_points
                    if node_id in execution_results
                }

                # Accumulate per-loop results (keyed to avoid overwriting)
                if self.max_loops > 1:
                    for node_id, output in execution_results.items():
                        all_loop_results[f"{node_id}_loop_{loop}"] = (
                            output
                        )

            # Build final return value
            total_execution_time = time.time() - run_start_time

            logger.info(
                f"GraphWorkflow execution completed: {len(execution_results)} agents executed across {self.max_loops} loop(s) in {total_execution_time:.3f}s"
            )

            if self.verbose:
                logger.debug(
                    f"Final execution results: {list(execution_results.keys())}"
                )

            # For single-loop (the common case), return results directly.
            # For multi-loop, merge the per-loop history with the final
            # loop's results so callers can access both.
            if self.max_loops > 1:
                all_loop_results.update(execution_results)
                return all_loop_results

            return execution_results

        except Exception as e:
            total_time = time.time() - run_start_time
            logger.exception(
                f"Error in GraphWorkflow.run after {total_time:.3f}s: {e}"
            )
            raise e

        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    def _fan_patterns(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Group edges both ways: ``(source -> targets, target -> sources)``.
        Both visualizers style and count fan-out/fan-in from these."""
        fan_out: Dict[str, List[str]] = {}
        fan_in: Dict[str, List[str]] = {}
        for edge in self.edges:
            fan_out.setdefault(edge.source, []).append(edge.target)
            fan_in.setdefault(edge.target, []).append(edge.source)
        return fan_out, fan_in

    @staticmethod
    def _branching(groups: Dict[str, List[str]]) -> int:
        """Count entries in a fan map that actually branch."""
        return sum(
            1 for members in groups.values() if len(members) > 1
        )

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
        # Sanitize here, in the path that is actually used. graphviz treats
        # the argument as a filesystem path, so a name containing "/" (or
        # any other separator) renders into a directory that does not exist
        # instead of producing a file.
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in (self.name or "GraphWorkflow")
        )
        output_path = f"{safe_name}_visualization_{str(uuid.uuid4())}"

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

            fan_out_nodes, fan_in_nodes = self._fan_patterns()

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

            # Render the graph
            output_file = dot.render(
                output_path, view=view, cleanup=True
            )

            # Show parallel processing summary
            if show_summary:
                fan_out_count = self._branching(fan_out_nodes)
                fan_in_count = self._branching(fan_in_nodes)
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
            fan_out_nodes, fan_in_nodes = self._fan_patterns()
            fan_out_count = self._branching(fan_out_nodes)
            fan_in_count = self._branching(fan_in_nodes)

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

    def clear_checkpoints(self, task: str) -> int:
        """
        Delete all checkpoint files written for a specific task.

        Call this after a workflow run completes successfully to reclaim disk
        space and prevent stale checkpoints from being picked up on future runs
        with the same task string.

        Args:
            task (str): The task string whose checkpoints should be removed.
                Must match the string passed to :meth:`run` exactly.

        Returns:
            int: Number of checkpoint files deleted.

        Raises:
            ValueError: If ``checkpoint_dir`` was not set on this workflow.

        Example::

            workflow.run("Analyse quarterly earnings")
            workflow.clear_checkpoints("Analyse quarterly earnings")
        """
        if not self.checkpoint_dir:
            raise ValueError(
                "checkpoint_dir is not set on this GraphWorkflow instance."
            )
        cp_dir = Path(self.checkpoint_dir)
        if not cp_dir.exists():
            return 0
        task_key = self._task_key(task)
        prefix = f"{task_key}_layer_"
        deleted = 0
        for cp_file in cp_dir.glob(f"{prefix}*.json"):
            try:
                cp_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(
                    f"Failed to delete checkpoint file {cp_file}: {e}"
                )
        if self.verbose:
            logger.info(
                f"Cleared {deleted} checkpoint file(s) for task key {task_key}"
            )
        return deleted

    # Persistence. Two shapes: shallow (topology only, each node names its
    # agent) and deep (agents embedded). Everything public picks one and
    # delegates to to_dict / _write_json / from_*.

    @staticmethod
    def _edge_payload(edge: "Edge") -> Dict[str, Any]:
        """Serialize one edge. Identical in both shapes."""
        return {
            "source": edge.source,
            "target": edge.target,
            "metadata": edge.metadata,
        }

    @staticmethod
    def _agent_payload(node: "Node") -> Dict[str, Any]:
        """Serialize a node's agent, falling back to a name/type stub so one
        bad agent cannot cost you the whole export."""
        agent = node.agent
        if hasattr(agent, "to_dict"):
            try:
                return agent.to_dict()
            except Exception as e:
                logger.warning(
                    f"Failed to serialize agent {node.id} to dict: {e}"
                )
                return {
                    "agent_name": getattr(
                        agent, "agent_name", str(agent)
                    ),
                    "serialization_error": str(e),
                    "agent_type": str(type(agent)),
                }
        return {
            "agent_name": getattr(agent, "agent_name", str(agent)),
            "agent_type": str(type(agent)),
            "serialization_method": "fallback_string",
        }

    def _node_payload(
        self, node_id: str, node: "Node", shallow: bool
    ) -> Dict[str, Any]:
        """Serialize one node in whichever shape was asked for."""
        if shallow:
            if node.type == NodeType.SUBGRAPH:
                return {
                    "id": node_id,
                    "type": "subgraph",
                    "spec": node.agent.to_dict(shallow=True),
                    "metadata": node.metadata,
                }
            return {
                "id": node_id,
                "agent_name": getattr(
                    node.agent, "agent_name", node_id
                ),
                "metadata": node.metadata,
            }

        return {
            "id": node.id,
            # ``.value``, not ``str(node.type)``: the latter renders as
            # "NodeType.AGENT", which NodeType() then refuses to parse back.
            "type": node.type.value,
            "metadata": node.metadata,
            "agent": self._agent_payload(node),
        }

    def to_dict(
        self,
        shallow: bool = True,
        include_conversation: bool = False,
        include_runtime_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Serialize the workflow to a dict. The single builder behind
        :meth:`to_spec`, :meth:`to_json`, :meth:`save` and friends.

        Args:
            shallow: Topology only — each node names its agent, subgraphs
                recurse, output is sorted and diffable. Rebuild with
                :meth:`from_topology_spec`. When ``False``, embed the
                serialized agents plus export metadata and metrics.
            include_conversation: Deep shape only. Embed conversation history.
            include_runtime_state: Deep shape only. Embed compilation state.

        Returns:
            The serialized workflow.
        """
        if shallow:
            # Sorted for deterministic output — two equivalent workflows
            # built in different insertion orders produce identical specs.
            return {
                "name": self.name,
                "description": self.description,
                "max_loops": self.max_loops,
                "nodes": [
                    self._node_payload(node_id, node, shallow=True)
                    for node_id, node in sorted(self.nodes.items())
                ],
                "edges": [
                    self._edge_payload(e)
                    for e in sorted(
                        self.edges, key=lambda e: (e.source, e.target)
                    )
                ],
                "entry_points": sorted(self.entry_points),
                "end_points": sorted(self.end_points),
            }

        data: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "export_timestamp": time.time(),
            "export_date": time.strftime(
                "%Y-%m-%d %H:%M:%S UTC", time.gmtime()
            ),
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [
                self._node_payload(node_id, node, shallow=False)
                for node_id, node in self.nodes.items()
            ],
            "edges": [self._edge_payload(e) for e in self.edges],
            "entry_points": self.entry_points,
            "end_points": self.end_points,
            "max_loops": self.max_loops,
            "auto_compile": self.auto_compile,
            "verbose": self.verbose,
            "task": self.task,
            "max_workers": self._max_workers,
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

        if include_conversation and self.conversation:
            try:
                if hasattr(self.conversation, "to_dict"):
                    data["conversation"] = self.conversation.to_dict()
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
                data["conversation"] = {"serialization_error": str(e)}

        if include_runtime_state:
            data["runtime_state"] = {
                "is_compiled": self._compiled,
                "compilation_timestamp": self._compilation_timestamp,
                "sorted_layers": (
                    self._sorted_layers if self._compiled else None
                ),
                "compilation_cache_valid": self._compiled,
                "time_since_compilation": (
                    time.time() - self._compilation_timestamp
                    if self._compilation_timestamp
                    else None
                ),
            }

        return data

    def to_spec(self) -> Dict[str, Any]:
        """
        The topology spec: agent names only, so it can be committed and diffed
        without agent internals. ``to_dict(shallow=True)``.

        Example::

            spec = workflow.to_spec()
            rebuilt = GraphWorkflow.from_topology_spec(spec, agent_registry)
        """
        return self.to_dict(shallow=True)

    def to_json(
        self,
        fast: bool = True,
        include_conversation: bool = False,
        include_runtime_state: bool = False,
    ) -> str:
        """
        Serialize to a JSON string with agents embedded.

        Args:
            fast: Ignored; kept for backward compatibility.
            include_conversation: Include conversation history.
            include_runtime_state: Include compilation state.

        Returns:
            JSON representation of the workflow.
        """
        try:
            result = json.dumps(
                self.to_dict(
                    shallow=False,
                    include_conversation=include_conversation,
                    include_runtime_state=include_runtime_state,
                ),
                indent=2,
                default=str,
            )
            if self.verbose:
                logger.success(
                    f"Serialized GraphWorkflow to JSON ({len(result)} characters, "
                    f"{len(self.nodes)} nodes, {len(self.edges)} edges)"
                )
            return result
        except Exception as e:
            logger.exception(f"Error in GraphWorkflow.to_json: {e}")
            raise e

    def _write_json(
        self,
        path: str,
        payload: Dict[str, Any],
        overwrite: bool = True,
        ensure_extension: bool = False,
    ) -> str:
        """
        Write ``payload`` as JSON, creating parent directories.

        Args:
            path: Destination path.
            payload: The dict to write.
            overwrite: When ``False``, refuse to replace an existing file.
            ensure_extension: Append ``.json`` when ``path`` lacks it.

        Returns:
            The path actually written.

        Raises:
            FileExistsError: If the file exists and ``overwrite`` is ``False``.
        """
        if ensure_extension and not path.endswith(".json"):
            path += ".json"

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"File {path} already exists. Set overwrite=True to replace it."
            )

        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        if self.verbose:
            logger.success(
                f"GraphWorkflow saved to {path} "
                f"({os.path.getsize(path):,} bytes)"
            )
        return path

    def save(
        self,
        path: str,
        shallow: bool = True,
        overwrite: bool = True,
        include_conversation: bool = False,
        include_runtime_state: bool = False,
    ) -> str:
        """
        Write the workflow to a JSON file. Pair with :meth:`load`.

        Args:
            path: Destination path.
            shallow: Topology only (default) or embed agents. See :meth:`to_dict`.
            overwrite: When ``False``, refuse to replace an existing file.
            include_conversation: Deep shape only.
            include_runtime_state: Deep shape only.

        Returns:
            The path actually written.
        """
        return self._write_json(
            path,
            self.to_dict(
                shallow=shallow,
                include_conversation=include_conversation,
                include_runtime_state=include_runtime_state,
            ),
            overwrite=overwrite,
        )

    def save_spec(self, path: str) -> None:
        """Write :meth:`to_spec` to ``path`` — the form to commit. Rebuild with
        :meth:`from_topology_spec`."""
        self.save(path, shallow=True)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_node_type(raw: Any) -> "NodeType":
        """Parse a node type. Older exports wrote "NodeType.AGENT", which
        NodeType() rejects, so accept both spellings."""
        if isinstance(raw, NodeType):
            return raw
        text = str(raw)
        if text.startswith("NodeType."):
            text = text.split(".", 1)[1].lower()
        try:
            return NodeType(text)
        except ValueError:
            return NodeType.AGENT

    @classmethod
    def _edges_from_payload(
        cls, payload: Dict[str, Any]
    ) -> List["Edge"]:
        """Rebuild edge objects from either serialized shape."""
        return [
            Edge(
                source=e["source"],
                target=e["target"],
                metadata=e.get("metadata") or {},
            )
            for e in payload.get("edges", [])
        ]

    @classmethod
    def from_topology_spec(
        cls,
        spec: Dict[str, Any],
        agent_registry: Dict[str, "Agent"],
        **kwargs: Any,
    ) -> "GraphWorkflow":
        """
        Rebuild from a topology spec: the spec says how nodes connect, the
        registry supplies the live agents. Counterpart to :meth:`to_spec`.

        Args:
            spec: A spec as returned by :meth:`to_spec`.
            agent_registry: ``agent_name -> Agent``. Every agent named in the
                spec must appear here.
            **kwargs: Forwarded to the constructor.

        Returns:
            The reconstructed workflow.

        Raises:
            ValueError: If the spec is malformed or an agent name is missing.
        """
        if not isinstance(spec, dict):
            raise ValueError(
                f"spec must be a dict, got {type(spec).__name__}"
            )
        if "nodes" not in spec:
            raise ValueError("spec is missing required key 'nodes'")

        # Subgraph nodes carry a nested "spec" instead of an "agent_name".
        for i, n in enumerate(spec.get("nodes", [])):
            if "id" not in n:
                raise ValueError(
                    f"Node at index {i} is missing required key 'id'"
                )
            if n.get("type") != "subgraph" and "agent_name" not in n:
                raise ValueError(
                    f"Node at index {i} is missing required key 'agent_name'"
                )
            if n.get("type") == "subgraph" and "spec" not in n:
                raise ValueError(
                    f"Subgraph node at index {i} is missing required key 'spec'"
                )

        for i, e in enumerate(spec.get("edges", [])):
            for key in ("source", "target"):
                if key not in e:
                    raise ValueError(
                        f"Edge at index {i} is missing required key '{key}'"
                    )

        missing = [
            n["agent_name"]
            for n in spec["nodes"]
            if n.get("type") != "subgraph"
            and n["agent_name"] not in agent_registry
        ]
        if missing:
            raise ValueError(
                f"The following agent names are referenced in the spec but not "
                f"found in agent_registry: {missing}"
            )

        nodes = {}
        for n in spec["nodes"]:
            if n.get("type") == "subgraph":
                inner_wf = cls.from_topology_spec(
                    n["spec"], agent_registry, **kwargs
                )
                nodes[n["id"]] = Node.from_subgraph(
                    inner_wf,
                    node_id=n["id"],
                    metadata=n.get("metadata") or {},
                )
            else:
                nodes[n["id"]] = Node(
                    id=n["id"],
                    agent=agent_registry[n["agent_name"]],
                    metadata=n.get("metadata") or {},
                )

        return cls(
            name=spec.get("name", "Loaded-Workflow"),
            description=spec.get("description", ""),
            max_loops=spec.get("max_loops", 1),
            nodes=nodes,
            edges=cls._edges_from_payload(spec),
            entry_points=spec.get("entry_points") or [],
            end_points=spec.get("end_points") or [],
            **kwargs,
        )

    @classmethod
    def from_json(
        cls,
        json_str: str,
        restore_runtime_state: bool = False,
        agent_registry: Optional[Dict[str, "Agent"]] = None,
    ) -> "GraphWorkflow":
        """
        Rebuild from a deep export. Nodes keep their exported ids so edges
        resolve; agents come back as dicts unless a registry rehydrates them.

        Args:
            json_str: JSON produced by :meth:`to_json`.
            restore_runtime_state: Restore compilation state.
            agent_registry: Optional ``agent_name -> Agent`` to wire nodes to.

        Returns:
            The reconstructed workflow.

        Raises:
            ValueError: If ``json_str`` is not valid JSON.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON format in GraphWorkflow.from_json: {e}"
            )
            raise ValueError(f"Invalid JSON format: {e}")

        try:
            schema_version = data.get("schema_version", "legacy")

            nodes = {}
            for n in data.get("nodes", []):
                try:
                    agent = n.get("agent")
                    if isinstance(agent, dict):
                        name = agent.get("agent_name")
                        if agent_registry and name in agent_registry:
                            agent = agent_registry[name]
                        elif "serialization_error" in agent:
                            logger.warning(
                                f"Node {n['id']} was exported with an agent "
                                f"serialization error: {agent['serialization_error']}"
                            )
                    nodes[n["id"]] = Node(
                        id=n["id"],
                        type=cls._parse_node_type(n.get("type")),
                        agent=agent,
                        metadata=n.get("metadata") or {},
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize node {n.get('id', 'unknown')}: {e}"
                    )

            params = {
                k: v
                for k, v in {
                    "id": data.get("id"),
                    "name": data.get("name", "Loaded-Workflow"),
                    "description": data.get(
                        "description", "Workflow loaded from JSON"
                    ),
                    "max_loops": data.get("max_loops", 1),
                    "task": data.get("task"),
                    "auto_compile": data.get("auto_compile", True),
                    "verbose": data.get("verbose", False),
                }.items()
                if v is not None
            }

            result = cls(
                nodes=nodes,
                edges=cls._edges_from_payload(data),
                entry_points=data.get("entry_points") or [],
                end_points=data.get("end_points") or [],
                **params,
            )

            if "max_workers" in data:
                result._max_workers = data["max_workers"]

            if data.get("conversation"):
                try:
                    from swarms.structs.conversation import (
                        Conversation,
                    )

                    conv_data = data["conversation"]
                    if (
                        isinstance(conv_data, dict)
                        and "history" in conv_data
                    ):
                        conv = Conversation()
                        conv.history = conv_data["history"]
                        result.conversation = conv
                    else:
                        logger.warning(
                            "Conversation data present but in unrecognized format"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to restore conversation: {e}"
                    )

            if restore_runtime_state and "runtime_state" in data:
                try:
                    runtime_state = data["runtime_state"]
                    if runtime_state.get("is_compiled", False):
                        result._compiled = True
                        result._compilation_timestamp = (
                            runtime_state.get("compilation_timestamp")
                        )
                        result._sorted_layers = (
                            runtime_state.get("sorted_layers") or []
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to restore runtime state: {e}"
                    )

            logger.success(
                f"Loaded GraphWorkflow: {len(nodes)} nodes, "
                f"{len(result.edges)} edges, schema_version: {schema_version}"
            )
            return result

        except Exception as e:
            logger.exception(f"Error in GraphWorkflow.from_json: {e}")
            raise e

    @classmethod
    def load(
        cls,
        path: str,
        agent_registry: Optional[Dict[str, "Agent"]] = None,
        restore_runtime_state: bool = False,
        **kwargs: Any,
    ) -> "GraphWorkflow":
        """
        Read back a file written by :meth:`save`, detecting which shape it
        holds so one call handles both.

        Args:
            path: Path to the JSON file.
            agent_registry: ``agent_name -> Agent``. Required for a shallow
                spec, optional for a deep export.
            restore_runtime_state: Deep exports only.
            **kwargs: Forwarded to the constructor for shallow specs.

        Returns:
            The reconstructed workflow.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If a shallow spec is loaded without a registry.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Workflow file not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        payload = json.loads(raw)
        # A shallow spec names its agents; a deep export embeds them.
        is_shallow = any(
            "agent_name" in n or n.get("type") == "subgraph"
            for n in payload.get("nodes", [])
        )

        if is_shallow:
            if agent_registry is None:
                raise ValueError(
                    f"{path} holds a topology spec, which stores agent names "
                    f"rather than agents. Pass agent_registry={{name: agent}} "
                    f"to resolve them."
                )
            return cls.from_topology_spec(
                payload, agent_registry, **kwargs
            )

        return cls.from_json(
            raw,
            restore_runtime_state=restore_runtime_state,
            agent_registry=agent_registry,
        )

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
        :meth:`save` with agents embedded, plus a ``.json`` extension and a
        refusal to clobber by default.

        Args:
            filepath: Path to save the JSON file.
            include_conversation: Include conversation history.
            include_runtime_state: Include compilation state.
            overwrite: Whether to replace an existing file.

        Returns:
            Path to the saved file.

        Raises:
            FileExistsError: If the file exists and ``overwrite`` is ``False``.
        """
        try:
            return self._write_json(
                filepath,
                self.to_dict(
                    shallow=False,
                    include_conversation=include_conversation,
                    include_runtime_state=include_runtime_state,
                ),
                overwrite=overwrite,
                ensure_extension=True,
            )
        except FileExistsError:
            raise
        except Exception as e:
            logger.exception(
                f"Failed to save GraphWorkflow to {filepath}: {e}"
            )
            raise e

    @classmethod
    def load_from_file(
        cls, filepath: str, restore_runtime_state: bool = False
    ) -> "GraphWorkflow":
        """:meth:`load` for files written by :meth:`save_to_file`. Raises
        ``FileNotFoundError`` if ``filepath`` does not exist."""
        return cls.load(
            filepath, restore_runtime_state=restore_runtime_state
        )

    def validate(
        self,
        auto_fix: bool = False,
        raise_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate the workflow structure, checking for potential issues such as isolated nodes,
        cyclic dependencies, unreachable nodes, and missing entry/end points.

        Runs the same checks as :meth:`compile` via :meth:`_structural_checks`,
        adding reporting and auto-repair.

        Args:
            auto_fix (bool): Whether to automatically fix simple issues such as
                auto-setting missing entry/exit points and adding unreachable
                nodes to entry points.
            raise_on_error (bool): When ``True``, raise a ``ValueError`` if
                validation determines the workflow is invalid (i.e.
                ``result["is_valid"]`` is ``False``).  Defaults to ``False``
                so that existing callers that inspect the returned dict are
                unaffected.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - ``is_valid`` (bool) — overall validity flag.
                - ``errors`` (List[str]) — fatal problems that prevent execution.
                - ``warnings`` (List[str]) — non-fatal issues worth addressing.
                - ``fixed`` (List[str]) — issues resolved when ``auto_fix=True``.
                - ``cycles`` (List[List[str]]) — detected cycles, if any.

        Raises:
            ValueError: If ``raise_on_error=True`` and the workflow is invalid.
        """
        if self.verbose:
            logger.debug(
                f"Validating GraphWorkflow structure (auto_fix={auto_fix})"
            )

        result: Dict[str, Any] = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "fixed": [],
        }

        try:
            if not self.nodes:
                result["errors"].append("Workflow has no nodes")
                result["is_valid"] = False
                return result

            # Missing entry/end points are repaired before the reachability
            # checks run, so auto_fix does not report nodes as unreachable
            # from a set of entry points it is about to populate.
            if not self.entry_points:
                result["warnings"].append("No entry points defined")
                if auto_fix:
                    self.auto_set_entry_points()
                    result["fixed"].append("Auto-set entry points")

            if not self.end_points:
                result["warnings"].append("No end points defined")
                if auto_fix:
                    self.auto_set_end_points()
                    result["fixed"].append("Auto-set end points")

            succ, pred = self.graph_backend.adjacency()
            errors, warnings, details = self._structural_checks(
                succ, pred, enumerate_cycles=True
            )
            result["errors"].extend(errors)
            result["warnings"].extend(warnings)
            if details["cycles"]:
                result["cycles"] = details["cycles"]
            if errors:
                result["is_valid"] = False

            if auto_fix and details["unreachable"]:
                self.set_entry_points(
                    self.entry_points + list(details["unreachable"])
                )
                result["fixed"].append(
                    f"Added {len(details['unreachable'])} unreachable "
                    f"nodes to entry points"
                )

            if auto_fix and details["dead_ends"]:
                self.set_end_points(
                    self.end_points + list(details["dead_ends"])
                )
                result["fixed"].append(
                    f"Added {len(details['dead_ends'])} dead-end nodes "
                    f"to exit points"
                )

            has_serious_warnings = any(
                "cycle" in warning.lower()
                or "unreachable" in warning.lower()
                for warning in result["warnings"]
            )

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

            if raise_on_error and not result["is_valid"]:
                all_issues = result["errors"] + result["warnings"]
                raise ValueError(
                    "GraphWorkflow validation failed:\n"
                    + "\n".join(f"  - {i}" for i in all_issues)
                )

            return result
        except ValueError:
            raise
        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(str(e))
            logger.exception(f"Error during workflow validation: {e}")
            if raise_on_error:
                raise ValueError(
                    f"GraphWorkflow validation failed: {e}"
                ) from e
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
