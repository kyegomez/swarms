"""Graph backend implementations for different graph libraries."""
from typing import Any, Dict, Iterator, List, Set
import networkx as nx

try:
    import rustworkx as rx
    RUSTWORKX_AVAILABLE = True
except ImportError:
    RUSTWORKX_AVAILABLE = False
    rx = None

from swarms.utils.loguru_logger import initialize_logger
logger = initialize_logger(log_folder="graph_workflow")

class GraphBackend:
    """Abstract base class for graph backends."""
    def add_node(self, node_id: str, **attrs) -> None: raise NotImplementedError
    def add_edge(self, source: str, target: str, **attrs) -> None: raise NotImplementedError
    def in_degree(self, node_id: str) -> int: raise NotImplementedError
    def out_degree(self, node_id: str) -> int: raise NotImplementedError
    def predecessors(self, node_id: str) -> Iterator[str]: raise NotImplementedError
    def reverse(self) -> "GraphBackend": raise NotImplementedError
    def topological_generations(self) -> List[List[str]]: raise NotImplementedError
    def simple_cycles(self) -> List[List[str]]: raise NotImplementedError
    def descendants(self, node_id: str) -> Set[str]: raise NotImplementedError

class NetworkXBackend(GraphBackend):
    """NetworkX backend implementation."""
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, **attrs) -> None: self.graph.add_node(node_id, **attrs)
    def add_edge(self, source: str, target: str, **attrs) -> None: self.graph.add_edge(source, target, **attrs)
    def in_degree(self, node_id: str) -> int: return self.graph.in_degree(node_id)
    def out_degree(self, node_id: str) -> int: return self.graph.out_degree(node_id)
    def predecessors(self, node_id: str) -> Iterator[str]: return self.graph.predecessors(node_id)

    def reverse(self) -> "NetworkXBackend":
        reversed_backend = NetworkXBackend()
        reversed_backend.graph = self.graph.reverse()
        return reversed_backend

    def topological_generations(self) -> List[List[str]]: return list(nx.topological_generations(self.graph))
    def simple_cycles(self) -> List[List[str]]: return list(nx.simple_cycles(self.graph))
    def descendants(self, node_id: str) -> Set[str]: return nx.descendants(self.graph, node_id)

class RustworkxBackend(GraphBackend):
    """Rustworkx backend implementation."""
    def __init__(self):
        if not RUSTWORKX_AVAILABLE:
            raise ImportError("rustworkx is not installed. Install it with: pip install rustworkx")
        self.graph = rx.PyDiGraph()
        self._node_id_to_index: Dict[str, int] = {}
        self._index_to_node_id: Dict[int, str] = {}

    def _get_or_create_node_index(self, node_id: str) -> int:
        if node_id not in self._node_id_to_index:
            node_index = self.graph.add_node(node_id)
            self._node_id_to_index[node_id] = node_index
            self._index_to_node_id[node_index] = node_id
        return self._node_id_to_index[node_id]

    def add_node(self, node_id: str, **attrs) -> None:
        if node_id not in self._node_id_to_index:
            node_data = {"node_id": node_id, **attrs}
            node_index = self.graph.add_node(node_data)
            self._node_id_to_index[node_id] = node_index
            self._index_to_node_id[node_index] = node_id
        else:
            node_index = self._node_id_to_index[node_id]
            node_data = self.graph[node_index]
            if isinstance(node_data, dict):
                node_data.update(attrs)
            else:
                self.graph[node_index] = {"node_id": node_id, **attrs}

    def add_edge(self, source: str, target: str, **attrs) -> None:
        source_idx = self._get_or_create_node_index(source)
        target_idx = self._get_or_create_node_index(target)
        edge_data = attrs if attrs else None
        self.graph.add_edge(source_idx, target_idx, edge_data)

    def in_degree(self, node_id: str) -> int:
        if node_id not in self._node_id_to_index: return 0
        return self.graph.in_degree(self._node_id_to_index[node_id])

    def out_degree(self, node_id: str) -> int:
        if node_id not in self._node_id_to_index: return 0
        return self.graph.out_degree(self._node_id_to_index[node_id])

    def predecessors(self, node_id: str) -> Iterator[str]:
        if node_id not in self._node_id_to_index: return iter([])
        target_index = self._node_id_to_index[node_id]
        result = []
        for edge in self.graph.edge_list():
            if edge[1] == target_index:
                result.append(self._index_to_node_id[edge[0]])
        return iter(result)

    def reverse(self) -> "RustworkxBackend":
        reversed_backend = RustworkxBackend()
        reversed_backend.graph = self.graph.copy()
        reversed_backend.graph.reverse()
        reversed_backend._node_id_to_index = self._node_id_to_index.copy()
        reversed_backend._index_to_node_id = self._index_to_node_id.copy()
        return reversed_backend

    def topological_generations(self) -> List[List[str]]:
        try:
            all_indices = list(self._node_id_to_index.values())
            if not all_indices: return []

            layers = []
            remaining = set(all_indices)
            processed = set()

            while remaining:
                layer = []
                nodes_to_add = []
                for idx in list(remaining):
                    preds = []
                    for e in self.graph.edge_list():
                        if e[1] == idx: preds.append(e[0])
                    
                    if all((p in processed) or (p == idx) for p in preds):
                        nodes_to_add.append(idx)
                        
                for idx in nodes_to_add:
                    processed.add(idx)
                    remaining.remove(idx)
                    layer.append(self._index_to_node_id[idx])
                    
                if not layer: break 
                layers.append(layer)

            if remaining:
                layers.append([self._index_to_node_id[idx] for idx in remaining])

            return layers if layers else [[self._index_to_node_id[idx] for idx in all_indices]]
        except Exception as e:
            logger.warning(f"Error in rustworkx topological_generations: {e}, falling back to simple approach")
            return [[node_id for node_id in self._node_id_to_index.keys()]]

    def simple_cycles(self) -> List[List[str]]:
        try:
            nx_graph = nx.DiGraph()
            for node_id in self._node_id_to_index.keys():
                nx_graph.add_node(node_id)
            for edge in self.graph.edge_list():
                source_id = self._index_to_node_id[edge[0]]
                target_id = self._index_to_node_id[edge[1]]
                nx_graph.add_edge(source_id, target_id)
            return list(nx.simple_cycles(nx_graph))
        except Exception as e:
            logger.warning(f"Error in rustworkx simple_cycles: {e}, returning empty list")
            return []

    def descendants(self, node_id: str) -> Set[str]:
        if node_id not in self._node_id_to_index:
            return set()
        node_index = self._node_id_to_index[node_id]
        descendants = set()
        queue = [node_index]
        visited = {node_index}

        while queue:
            current_idx = queue.pop(0)
            succ_data = self.graph.successors(current_idx)
            for succ in succ_data:
                if isinstance(succ, dict):
                    succ_idx = self._node_id_to_index[succ["node_id"]]
                elif isinstance(succ, int):
                    succ_idx = succ
                else:
                    continue

                if succ_idx not in visited:
                    visited.add(succ_idx)
                    queue.append(succ_idx)
                    descendants.add(self._index_to_node_id[succ_idx])

        return descendants
