from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from enum import Enum
from uuid import uuid4, UUID
import math
import random
import numpy as np
from collections import defaultdict, deque, Counter

from loguru import logger


class _NodeType(str, Enum):
    PROBLEM = "problem"
    HYPOTHESIS = "hypothesis"
    SUBPROBLEM = "subproblem"
    INTERMEDIATE = "intermediate"
    RESULT = "result"
    FINAL = "final"
    CORRECTION = "correction"


class _EdgeRelation(str, Enum):
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    SUPPORTS = "supports"
    CORRECTS = "corrects"
    CRITICIZES = "criticizes"
    MERGES = "merges"
    EXTENDS = "extends"


class _GraphOperation(str, Enum):
    EXPAND = "expand"
    MERGE = "merge"
    REFINE = "refine"
    ADD_EDGE = "add_edge"
    FEEDBACK = "feedback"
    STOP = "stop"
    QUANTUM = "quantum"  # Quantum-inspired graph operations


class GraphInformationTheory:
    @staticmethod
    def graph_entropy(graph_probs: List[float]) -> float:
        if not graph_probs:
            return 0.0
        
        total = sum(graph_probs)
        if total == 0:
            return 0.0
        
        normalized = [p / total for p in graph_probs]
        
        h = 0.0
        for p in normalized:
            if p > 0:
                h -= p * math.log2(p)
        
        return h
    
    @staticmethod
    def mutual_information(
        prior_entropy: float,
        conditional_entropy: float
    ) -> float:
        return prior_entropy - conditional_entropy
    
    @staticmethod
    def node_information_gain(
        entropy_before: float,
        entropy_after: float
    ) -> float:
        return entropy_before - entropy_after
    
    @staticmethod
    def graph_complexity(
        num_nodes: int,
        num_edges: int,
        total_text_length: int
    ) -> float:
        if num_nodes == 0:
            return 0.0
        
        node_complexity = num_nodes * math.log2(max(1, num_nodes))
        edge_complexity = num_edges * math.log2(max(1, num_edges))
        text_complexity = total_text_length
        
        return node_complexity + edge_complexity + text_complexity


class QuantumGraphOperations:
    @staticmethod
    def calculate_graph_amplitudes(graph_probs: List[float]) -> List[float]:
        return [math.sqrt(max(0.0, p)) for p in graph_probs]
    
    @staticmethod
    def quantum_graph_measurement(
        graphs: List[Any],
        answers: List[str],
        graph_probs: Optional[List[float]] = None
    ) -> Tuple[str, float]:
        if not graphs or not answers:
            return "", 0.0
        
        if graph_probs is None:
            graph_probs = [1.0 / len(graphs)] * len(graphs)
        
        amplitudes = QuantumGraphOperations.calculate_graph_amplitudes(graph_probs)
        
        answer_amplitudes: Dict[str, float] = {}
        for answer, amp in zip(answers, amplitudes):
            normalized_answer = answer.lower().strip()
            answer_amplitudes[normalized_answer] = answer_amplitudes.get(normalized_answer, 0.0) + amp
        
        answer_probs = {ans: amp ** 2 for ans, amp in answer_amplitudes.items()}
        
        total = sum(answer_probs.values())
        if total > 0:
            answer_probs = {ans: prob / total for ans, prob in answer_probs.items()}
        
        if answer_probs:
            best_answer = max(answer_probs.items(), key=lambda x: x[1])
            return best_answer[0], best_answer[1]
        
        return "", 0.0
    
    @staticmethod
    def quantum_graph_sampling(
        graphs: List[Any],
        graph_probs: List[float],
        num_samples: int = 1
    ) -> List[Any]:
        if not graphs:
            return []
        
        total = sum(graph_probs)
        if total == 0:
            graph_probs = [1.0 / len(graphs)] * len(graphs)
        else:
            graph_probs = [p / total for p in graph_probs]
        
        amplitudes = QuantumGraphOperations.calculate_graph_amplitudes(graph_probs)
        probs = [amp ** 2 for amp in amplitudes]
        
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        
        sampled_indices = random.choices(
            range(len(graphs)),
            weights=probs,
            k=num_samples
        )
        
        return [graphs[i] for i in sampled_indices]


class GraphEnergyFunction:
    @staticmethod
    def calculate_graph_energy(graph_logprob: float) -> float:
        return -graph_logprob
    
    @staticmethod
    def boltzmann_graph_weight(energy: float, temperature: float) -> float:
        if temperature <= 0:
            return 0.0 if energy > 0 else 1.0
        
        return math.exp(-energy / temperature)
    
    @staticmethod
    def graph_partition_function(
        graph_energies: List[float],
        temperature: float
    ) -> float:
        if temperature <= 0:
            return 1.0
        
        weights = [GraphEnergyFunction.boltzmann_graph_weight(e, temperature) for e in graph_energies]
        return sum(weights)
    
    @staticmethod
    def graph_free_energy(partition_function: float, temperature: float) -> float:
        if partition_function <= 0:
            return float('inf')
        
        if temperature <= 0:
            return 0.0
        
        return -temperature * math.log(partition_function)
    
    @staticmethod
    def graph_ensemble_average(
        graph_values: List[float],
        graph_energies: List[float],
        temperature: float
    ) -> float:
        if not graph_values or not graph_energies:
            return 0.0
        
        z = GraphEnergyFunction.graph_partition_function(graph_energies, temperature)
        if z <= 0:
            return 0.0
        
        weighted_sum = sum(
            val * GraphEnergyFunction.boltzmann_graph_weight(energy, temperature)
            for val, energy in zip(graph_values, graph_energies)
        )
        
        return weighted_sum / z


class SpectralGraphTheory:
    @staticmethod
    def compute_laplacian(adjacency_matrix: np.ndarray) -> np.ndarray:
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        laplacian = degree_matrix - adjacency_matrix
        return laplacian
    
    @staticmethod
    def compute_normalized_laplacian(adjacency_matrix: np.ndarray) -> np.ndarray:
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        degree_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix) + 1e-10))
        laplacian = SpectralGraphTheory.compute_laplacian(adjacency_matrix)
        normalized = degree_inv_sqrt @ laplacian @ degree_inv_sqrt
        return normalized
    
    @staticmethod
    def compute_spectrum(
        laplacian: np.ndarray,
        k: Optional[int] = None,
        which: str = "SM"
    ) -> Tuple[np.ndarray, np.ndarray]:
        if laplacian.size == 0:
            return np.array([]), np.array([])
        
        try:
            # Full eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            
            # Sort eigenvalues
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Select k eigenvalues if requested
            if k is not None and k < len(eigenvalues):
                if which == "SM":
                    # Smallest k eigenvalues
                    eigenvalues = eigenvalues[:k]
                    eigenvectors = eigenvectors[:, :k]
                elif which == "LM":
                    # Largest k eigenvalues
                    eigenvalues = eigenvalues[-k:]
                    eigenvectors = eigenvectors[:, -k:]
                elif which == "BE":
                    # Both ends: k//2 smallest and k//2 largest
                    k_small = k // 2
                    k_large = k - k_small
                    eigenvalues = np.concatenate([eigenvalues[:k_small], eigenvalues[-k_large:]])
                    eigenvectors = np.concatenate([eigenvectors[:, :k_small], eigenvectors[:, -k_large:]], axis=1)
            
            return eigenvalues, eigenvectors
            
        except np.linalg.LinAlgError as e:
            logger.warning(f"Eigendecomposition failed: {e}, using fallback")
            # Fallback: use SVD for singular matrices
            try:
                U, s, Vt = np.linalg.svd(laplacian, full_matrices=False)
                eigenvalues = s
                eigenvectors = U
                return eigenvalues, eigenvectors
            except Exception:
                return np.array([]), np.array([])
    
    @staticmethod
    def graph_fourier_transform(
        node_embeddings: np.ndarray,
        eigenvectors: np.ndarray
    ) -> np.ndarray:
        if eigenvectors.size == 0 or node_embeddings.size == 0:
            return np.array([])
        
        # F = Φ^T H
        fourier_coeffs = eigenvectors.T @ node_embeddings
        return fourier_coeffs


class GraphTopology:
    @staticmethod
    def degree_centrality(
        adjacency_matrix: np.ndarray,
        node_idx: int
    ) -> float:
        n = adjacency_matrix.shape[0]
        if n <= 1:
            return 0.0
        
        degree = np.sum(adjacency_matrix[node_idx, :])
        return degree / (n - 1)
    
    @staticmethod
    def clustering_coefficient(
        adjacency_matrix: np.ndarray,
        node_idx: int
    ) -> float:
        neighbors = np.where(adjacency_matrix[node_idx, :] > 0)[0]
        k_v = len(neighbors)
        
        if k_v < 2:
            return 0.0
        
        # Count edges among neighbors
        e_v = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if adjacency_matrix[neighbors[i], neighbors[j]] > 0:
                    e_v += 1
        
        return (2.0 * e_v) / (k_v * (k_v - 1))
    
    @staticmethod
    def shortest_path_length(
        adjacency_matrix: np.ndarray,
        source: int,
        target: int,
        weighted: bool = False
    ) -> float:
        n = adjacency_matrix.shape[0]
        if source == target:
            return 0.0
        
        if source < 0 or source >= n or target < 0 or target >= n:
            return float('inf')
        
        if weighted:
            # Dijkstra's algorithm for weighted shortest paths
            distances = np.full(n, np.inf)
            distances[source] = 0.0
            visited = np.zeros(n, dtype=bool)
            prev = np.full(n, -1, dtype=int)
            
            # Priority queue: (distance, node)
            import heapq
            pq = [(0.0, source)]
            
            while pq:
                current_dist, current = heapq.heappop(pq)
                
                if visited[current]:
                    continue
                
                visited[current] = True
                
                if current == target:
                    # Reconstruct path if needed
                    return float(current_dist)
                
                # Update neighbors
                for neighbor in range(n):
                    if adjacency_matrix[current, neighbor] > 0 and not visited[neighbor]:
                        edge_weight = adjacency_matrix[current, neighbor]
                        new_dist = current_dist + edge_weight
                        
                        if new_dist < distances[neighbor]:
                            distances[neighbor] = new_dist
                            prev[neighbor] = current
                            heapq.heappush(pq, (new_dist, neighbor))
            
            return float(distances[target]) if distances[target] != np.inf else float('inf')
        else:
            # BFS for unweighted shortest paths
            queue = deque([source])
            distances = {source: 0}
            
            while queue:
                current = queue.popleft()
                
                if current == target:
                    return float(distances[current])
                
                for neighbor in np.where(adjacency_matrix[current, :] > 0)[0]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            
            return float('inf')
    
    @staticmethod
    def all_pairs_shortest_paths(adjacency_matrix: np.ndarray) -> np.ndarray:
        n = adjacency_matrix.shape[0]
        if n == 0:
            return np.array([])
        
        # Initialize distance matrix
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0.0)
        
        # Set direct edges
        for i in range(n):
            for j in range(n):
                if adjacency_matrix[i, j] > 0:
                    dist[i, j] = adjacency_matrix[i, j]
        
        # Floyd-Warshall: d[i, j] = min(d[i, j], d[i, k] + d[k, j])
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] != np.inf and dist[k, j] != np.inf:
                        dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
        
        return dist
    
    @staticmethod
    def graph_diameter(adjacency_matrix: np.ndarray) -> float:
        n = adjacency_matrix.shape[0]
        if n <= 1:
            return 0.0
        
        # Use all-pairs shortest paths for efficiency
        dist_matrix = GraphTopology.all_pairs_shortest_paths(adjacency_matrix)
        
        # Find maximum distance (excluding inf and diagonal)
        valid_distances = dist_matrix[dist_matrix != np.inf]
        valid_distances = valid_distances[valid_distances > 0]  # Exclude diagonal
        
        if len(valid_distances) == 0:
            return float('inf')
        
        return float(np.max(valid_distances))
    
    @staticmethod
    def closeness_centrality(
        adjacency_matrix: np.ndarray,
        node_idx: int,
        normalized: bool = True
    ) -> float:
        n = adjacency_matrix.shape[0]
        if n <= 1:
            return 0.0
        
        if node_idx < 0 or node_idx >= n:
            return 0.0
        
        # Use all-pairs shortest paths for efficiency if computing for multiple nodes
        # For single node, use individual shortest path calculations
        total_distance = 0.0
        reachable = 0
        
        for u in range(n):
            if u != node_idx:
                distance = GraphTopology.shortest_path_length(adjacency_matrix, node_idx, u, weighted=False)
                if distance != float('inf'):
                    total_distance += distance
                    reachable += 1
        
        if reachable == 0 or total_distance == 0:
            return 0.0
        
        if normalized:
            return (n - 1) / total_distance
        else:
            return 1.0 / total_distance
    
    @staticmethod
    def betweenness_centrality(
        adjacency_matrix: np.ndarray,
        node_idx: int,
        normalized: bool = True
    ) -> float:
        n = adjacency_matrix.shape[0]
        if n <= 2:
            return 0.0
        
        if node_idx < 0 or node_idx >= n:
            return 0.0
        
        # Brandes' algorithm for betweenness centrality
        betweenness = 0.0
        
        for s in range(n):
            if s == node_idx:
                continue
            
            # BFS from source s
            stack = []
            pred = [[] for _ in range(n)]
            sigma = np.zeros(n)
            sigma[s] = 1.0
            dist = np.full(n, -1, dtype=int)
            dist[s] = 0
            
            queue = deque([s])
            
            while queue:
                v = queue.popleft()
                stack.append(v)
                
                for w in np.where(adjacency_matrix[v, :] > 0)[0]:
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            
            # Accumulation
            delta = np.zeros(n)
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    if w == node_idx:
                        betweenness += delta[w]
        
        if normalized:
            # Normalize by (n - 1)(n - 2) / 2 for directed graphs
            normalization = (n - 1) * (n - 2)
            if normalization > 0:
                betweenness /= normalization
        
        return float(betweenness)
    
    @staticmethod
    def eigenvector_centrality(
        adjacency_matrix: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        n = adjacency_matrix.shape[0]
        if n == 0:
            return np.array([])
        
        # Initialize with uniform values
        centrality = np.ones(n) / np.sqrt(n)
        
        for _ in range(max_iter):
            # Update: C' = A C
            centrality_new = adjacency_matrix @ centrality
            
            # Normalize
            norm = np.linalg.norm(centrality_new)
            if norm == 0:
                break
            
            centrality_new = centrality_new / norm
            
            # Check convergence
            if np.linalg.norm(centrality_new - centrality) < tol:
                break
            
            centrality = centrality_new
        
        return centrality
    
    @staticmethod
    def degree_distribution(adjacency_matrix: np.ndarray) -> Dict[int, float]:
        n = adjacency_matrix.shape[0]
        if n == 0:
            return {}
        
        degrees = np.sum(adjacency_matrix, axis=1).astype(int)
        degree_counts = Counter(degrees)
        
        distribution = {k: count / n for k, count in degree_counts.items()}
        return distribution


class MDPFormulation:
    @staticmethod
    def value_function_estimate(
        rewards: List[float],
        gamma: float = 0.99,
        use_gae: bool = False,
        lambda_gae: float = 0.95
    ) -> Union[float, np.ndarray]:
        if not rewards:
            return 0.0
        
        rewards_array = np.array(rewards)
        T = len(rewards)
        
        if use_gae:
            # Generalized Advantage Estimation
            # A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            # where δ_t = r_t + γV(s_{t+1}) - V(s_t)
            
            # For simplicity, assume V(s_t) = 0 for all states
            # In practice, you would use a value function approximator
            values = np.zeros(T + 1)
            advantages = np.zeros(T)
            
            # Compute TD errors
            td_errors = rewards_array + gamma * values[1:] - values[:-1]
            
            # Compute GAE
            gae = 0.0
            for t in reversed(range(T)):
                delta = td_errors[t]
                gae = delta + gamma * lambda_gae * gae
                advantages[t] = gae
            
            # Value estimate is sum of advantages
            return float(np.sum(advantages))
        else:
            # Standard discounted return
            value = 0.0
            for t, reward in enumerate(rewards):
                value += (gamma ** t) * reward
            
            return value
    
    @staticmethod
    def q_function_estimate(
        state_value: float,
        action_reward: float,
        next_state_value: float,
        gamma: float = 0.99
    ) -> float:
        return action_reward + gamma * next_state_value
    
    @staticmethod
    def advantage_function(
        q_value: float,
        value: float
    ) -> float:
        return q_value - value
    
    @staticmethod
    def policy_gradient_estimate(
        log_probs: List[float],
        rewards: List[float],
        gamma: float = 0.99
    ) -> float:
        if not log_probs or not rewards:
            return 0.0
        
        if len(log_probs) != len(rewards):
            return 0.0
        
        gradient = 0.0
        for t, (log_prob, reward) in enumerate(zip(log_probs, rewards)):
            discounted_reward = (gamma ** t) * reward
            gradient += log_prob * discounted_reward
        
        return gradient / len(log_probs)


class GraphMatching:
    @staticmethod
    def graph_edit_distance(
        graph1_nodes: List[str],
        graph1_edges: List[Tuple[int, int]],
        graph2_nodes: List[str],
        graph2_edges: List[Tuple[int, int]],
        node_sub_cost: Callable[[str, str], float] = None,
        edge_sub_cost: float = 1.0,
        node_ins_cost: float = 1.0,
        node_del_cost: float = 1.0
    ) -> float:
        if node_sub_cost is None:
            def node_sub_cost(s1: str, s2: str) -> float:
                # Simple edit distance between strings
                m, n = len(s1), len(s2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i-1] == s2[j-1]:
                            dp[i][j] = dp[i-1][j-1]
                        else:
                            dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                return dp[m][n] / max(m, n, 1)
        
        n1, n2 = len(graph1_nodes), len(graph2_nodes)
        
        # Build edge sets for faster lookup
        edges1 = set(graph1_edges)
        edges2 = set(graph2_edges)
        
        # Approximate edit distance using node matching
        # Full implementation would use more sophisticated algorithms
        total_cost = 0.0
        
        # Node substitution costs
        min_size = min(n1, n2)
        for i in range(min_size):
            cost = node_sub_cost(graph1_nodes[i], graph2_nodes[i])
            total_cost += cost
        
        # Node insertion/deletion costs
        if n1 > n2:
            total_cost += (n1 - n2) * node_ins_cost
        elif n2 > n1:
            total_cost += (n2 - n1) * node_del_cost
        
        # Edge costs (simplified)
        edge_diff = len(edges1.symmetric_difference(edges2))
        total_cost += edge_diff * edge_sub_cost
        
        return total_cost
    
    @staticmethod
    def weisfeiler_lehman_kernel(
        graph1_nodes: List[str],
        graph1_edges: List[Tuple[int, int]],
        graph2_nodes: List[str],
        graph2_edges: List[Tuple[int, int]],
        num_iterations: int = 3
    ) -> float:
        def wl_relabel(nodes: List[str], edges: List[Tuple[int, int]], iteration: int) -> List[str]:
            """Perform one iteration of WL relabeling."""
            n = len(nodes)
            new_labels = nodes.copy()
            
            # Build adjacency list
            adj_list = [[] for _ in range(n)]
            for u, v in edges:
                adj_list[u].append(v)
                adj_list[v].append(u)
            
            # Relabel based on neighbors
            for i in range(n):
                neighbor_labels = sorted([new_labels[j] for j in adj_list[i]])
                new_label = new_labels[i] + "," + ",".join(neighbor_labels)
                new_labels[i] = str(hash(new_label))
            
            return new_labels
        
        # Initial labels
        labels1 = graph1_nodes.copy()
        labels2 = graph2_nodes.copy()
        
        kernel_value = 0.0
        
        for k in range(num_iterations + 1):
            # Count matching labels
            label_counts1 = Counter(labels1)
            label_counts2 = Counter(labels2)
            
            # Intersection of label sets
            common_labels = set(label_counts1.keys()) & set(label_counts2.keys())
            for label in common_labels:
                kernel_value += label_counts1[label] * label_counts2[label]
            
            if k < num_iterations:
                # Relabel for next iteration
                labels1 = wl_relabel(labels1, graph1_edges, k)
                labels2 = wl_relabel(labels2, graph2_edges, k)
        
        return float(kernel_value)
    
    @staticmethod
    def graph_kernel(
        graph1_embedding: np.ndarray,
        graph2_embedding: np.ndarray,
        kernel_type: str = "rbf",
        gamma: float = 1.0
    ) -> float:
        if graph1_embedding.size == 0 or graph2_embedding.size == 0:
            return 0.0
        
        # Ensure same dimension
        min_dim = min(len(graph1_embedding), len(graph2_embedding))
        g1 = graph1_embedding[:min_dim]
        g2 = graph2_embedding[:min_dim]
        
        if kernel_type == "linear":
            # Linear kernel: K(x, y) = x^T y
            return float(np.dot(g1, g2))
        elif kernel_type == "polynomial":
            # Polynomial kernel: K(x, y) = (γ x^T y + 1)^d
            d = 2  # Degree
            return float((gamma * np.dot(g1, g2) + 1) ** d)
        elif kernel_type == "rbf":
            # RBF kernel: K(x, y) = exp(-γ ||x - y||²)
            diff = g1 - g2
            return float(np.exp(-gamma * np.dot(diff, diff)))
        else:
            return float(np.dot(g1, g2))  # Default to linear


class GraphNeuralNetwork:
    @staticmethod
    def gcn_layer(
        node_embeddings: np.ndarray,
        adjacency_matrix: np.ndarray,
        weight_matrix: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = False
    ) -> np.ndarray:
        n, d = node_embeddings.shape
        
        if n == 0 or d == 0:
            return node_embeddings
        
        # Add self-loops: A' = A + I
        adj_with_loops = adjacency_matrix + np.eye(n)
        
        # Compute degree matrix: D = diag(Σ_j A'_{ij})
        degree_matrix = np.diag(np.sum(adj_with_loops, axis=1))
        
        # Avoid division by zero
        degree_matrix = np.maximum(degree_matrix, 1e-10)
        
        # Normalize: Ã = D^{-1/2} A' D^{-1/2}
        degree_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
        normalized_adj = degree_inv_sqrt @ adj_with_loops @ degree_inv_sqrt
        
        # Initialize weight matrix if not provided
        if weight_matrix is None:
            # Use Xavier/Glorot initialization
            d_out = d
            weight_matrix = np.random.randn(d, d_out) * np.sqrt(2.0 / (d + d_out))
        else:
            d_out = weight_matrix.shape[1]
        
        # Message passing: H' = Ã H W
        h_next = normalized_adj @ node_embeddings @ weight_matrix
        
        # Add bias if provided
        if bias is not None:
            h_next = h_next + bias
        
        # Apply layer normalization if requested
        if use_layer_norm:
            mean = np.mean(h_next, axis=1, keepdims=True)
            std = np.std(h_next, axis=1, keepdims=True) + 1e-8
            h_next = (h_next - mean) / std
        
        # Apply activation function
        if activation == "relu":
            h_next = np.maximum(0, h_next)
        elif activation == "gelu":
            # GELU approximation: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
            h_next = h_next * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / math.pi) * (h_next + 0.044715 * h_next ** 3)))
        elif activation == "tanh":
            h_next = np.tanh(h_next)
        elif activation == "sigmoid":
            h_next = 1.0 / (1.0 + np.exp(-np.clip(h_next, -500, 500)))
        elif activation == "none":
            pass
        else:
            h_next = np.maximum(0, h_next)  # Default to ReLU
        
        # Apply dropout
        if dropout > 0.0 and dropout < 1.0:
            dropout_mask = np.random.binomial(1, 1.0 - dropout, h_next.shape)
            h_next = h_next * dropout_mask / (1.0 - dropout)
        
        return h_next
    
    @staticmethod
    def gat_layer(
        node_embeddings: np.ndarray,
        adjacency_matrix: np.ndarray,
        num_heads: int = 4,
        attention_dropout: float = 0.0
    ) -> np.ndarray:
        n, d = node_embeddings.shape
        
        if n == 0 or d == 0:
            return node_embeddings
        
        # Initialize attention mechanism parameters
        d_head = d // num_heads
        if d_head == 0:
            d_head = 1
            num_heads = d
        
        # Weight matrix for each head: W ∈ R^{d×d_head}
        W = np.random.randn(d, d_head * num_heads) * np.sqrt(2.0 / d)
        
        # Attention mechanism: a ∈ R^{2*d_head}
        a = np.random.randn(2 * d_head) * 0.01
        
        # Multi-head attention
        head_outputs = []
        
        for head in range(num_heads):
            W_head = W[:, head * d_head:(head + 1) * d_head]
            
            # Transform: h' = h W
            h_transformed = node_embeddings @ W_head
            
            # Compute attention scores for all pairs
            attention_scores = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if adjacency_matrix[i, j] > 0 or i == j:  # Include self-loops
                        # Concatenate: [W h_i || W h_j]
                        concat = np.concatenate([h_transformed[i], h_transformed[j]])
                        
                        # LeakyReLU(a^T [W h_i || W h_j])
                        score = np.dot(a, concat)
                        score = np.maximum(0.01 * score, score)  # LeakyReLU with α=0.01
                        
                        attention_scores[i, j] = score
            
            # Apply softmax to get attention weights
            # Mask out non-adjacent nodes
            mask = (adjacency_matrix > 0) | np.eye(n, dtype=bool)
            attention_scores = np.where(mask, attention_scores, -1e9)
            
            # Softmax: α_{ij} = exp(score_{ij}) / Σ_k exp(score_{ik})
            exp_scores = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
            attention_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)
            
            # Apply dropout to attention weights
            if attention_dropout > 0.0:
                dropout_mask = np.random.binomial(1, 1.0 - attention_dropout, attention_weights.shape)
                attention_weights = attention_weights * dropout_mask / (1.0 - attention_dropout)
            
            # Aggregate: h_i' = Σ_j α_{ij} W h_j
            h_head = attention_weights @ h_transformed
            
            head_outputs.append(h_head)
        
        # Concatenate all heads
        h_next = np.concatenate(head_outputs, axis=1)
        
        # Apply activation
        h_next = np.maximum(0, h_next)  # ReLU
        
        return h_next
    
    @staticmethod
    def graph_transformer_layer(
        node_embeddings: np.ndarray,
        adjacency_matrix: np.ndarray,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0
    ) -> np.ndarray:
        n, d = node_embeddings.shape
        
        if n == 0 or d == 0:
            return node_embeddings
        
        if ff_dim is None:
            ff_dim = 4 * d
        
        # Self-attention block
        h_attn = GraphNeuralNetwork._multi_head_attention(
            node_embeddings, node_embeddings, node_embeddings,
            adjacency_matrix, num_heads, dropout
        )
        
        # Residual connection and layer norm
        h_norm1 = GraphNeuralNetwork._layer_norm(node_embeddings + h_attn)
        
        # Feed-forward network
        h_ffn = GraphNeuralNetwork._feed_forward(h_norm1, ff_dim, dropout)
        
        # Residual connection and layer norm
        h_norm2 = GraphNeuralNetwork._layer_norm(h_norm1 + h_ffn)
        
        return h_norm2
    
    @staticmethod
    def _multi_head_attention(
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: np.ndarray,
        num_heads: int,
        dropout: float
    ) -> np.ndarray:
        n, d = query.shape
        d_head = d // num_heads
        
        if d_head == 0:
            d_head = 1
            num_heads = d
        
        # Split into heads and compute Q, K, V
        Q = query.reshape(n, num_heads, d_head)
        K = key.reshape(n, num_heads, d_head)
        V = value.reshape(n, num_heads, d_head)
        
        # Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V
        scores = np.einsum('nhd,mhd->nmh', Q, K) / np.sqrt(d_head)
        
        # Apply mask
        mask_expanded = np.expand_dims(mask, axis=2)
        scores = np.where(mask_expanded > 0, scores, -1e9)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attn_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)
        
        # Apply dropout
        if dropout > 0.0:
            dropout_mask = np.random.binomial(1, 1.0 - dropout, attn_weights.shape)
            attn_weights = attn_weights * dropout_mask / (1.0 - dropout)
        
        # Weighted sum
        h_attn = np.einsum('nmh,mhd->nhd', attn_weights, V)
        h_attn = h_attn.reshape(n, d)
        
        return h_attn
    
    @staticmethod
    def _feed_forward(
        x: np.ndarray,
        ff_dim: int,
        dropout: float
    ) -> np.ndarray:
        n, d = x.shape
        
        # First linear layer
        W1 = np.random.randn(d, ff_dim) * np.sqrt(2.0 / d)
        b1 = np.zeros(ff_dim)
        h1 = x @ W1 + b1
        h1 = np.maximum(0, h1)  # ReLU
        
        # Dropout
        if dropout > 0.0:
            dropout_mask = np.random.binomial(1, 1.0 - dropout, h1.shape)
            h1 = h1 * dropout_mask / (1.0 - dropout)
        
        # Second linear layer
        W2 = np.random.randn(ff_dim, d) * np.sqrt(2.0 / ff_dim)
        b2 = np.zeros(d)
        h2 = h1 @ W2 + b2
        
        return h2
    
    @staticmethod
    def _layer_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True) + eps
        normalized = (x - mean) / std
        # In practice, γ and β are learnable parameters (here we use γ=1, β=0)
        return normalized
    
    @staticmethod
    def graph_readout(
        node_embeddings: np.ndarray,
        method: str = "mean",
        attention_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if node_embeddings.size == 0:
            return np.array([])
        
        n, d = node_embeddings.shape
        
        if method == "mean":
            return np.mean(node_embeddings, axis=0)
        elif method == "max":
            return np.max(node_embeddings, axis=0)
        elif method == "sum":
            return np.sum(node_embeddings, axis=0)
        elif method == "attention":
            # Attention-based readout: h_G = Σ_v α_v h_v
            if attention_weights is None:
                # Learn attention weights
                attention_vector = np.random.randn(d) * 0.01
                scores = node_embeddings @ attention_vector
                scores = np.exp(scores - np.max(scores))
                attention_weights = scores / (np.sum(scores) + 1e-8)
            
            return np.sum(node_embeddings * attention_weights.reshape(-1, 1), axis=0)
        else:
            return np.mean(node_embeddings, axis=0)  # Default to mean


@dataclass
class _ThoughtNode:
    
    id: UUID
    text: str
    node_type: "_NodeType"
    parents: Set[UUID] = field(default_factory=set)
    children: Set[UUID] = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_ancestors(self, graph: "_ThoughtGraph") -> Set[UUID]:
        ancestors = set()
        queue = deque([self.id])
        
        while queue:
            node_id = queue.popleft()
            if node_id in ancestors:
                continue
            ancestors.add(node_id)
            
            node = graph.nodes.get(node_id)
            if node:
                for parent_id in node.parents:
                    if parent_id not in ancestors:
                        queue.append(parent_id)
        
        ancestors.discard(self.id)  # Remove self
        return ancestors
    
    def get_descendants(self, graph: "_ThoughtGraph") -> Set[UUID]:
        descendants = set()
        queue = deque([self.id])
        
        while queue:
            node_id = queue.popleft()
            if node_id in descendants:
                continue
            descendants.add(node_id)
            
            node = graph.nodes.get(node_id)
            if node:
                for child_id in node.children:
                    if child_id not in descendants:
                        queue.append(child_id)
        
        descendants.discard(self.id)  # Remove self
        return descendants


@dataclass
class _ThoughtEdge:
    
    source: UUID
    target: UUID
    relation: "_EdgeRelation"
    weight: float = 1.0


@dataclass
class _ThoughtGraph:
    
    nodes: Dict[UUID, "_ThoughtNode"] = field(default_factory=dict)
    edges: List["_ThoughtEdge"] = field(default_factory=list)
    root_id: Optional[UUID] = None
    
    def add_node(
        self,
        node_id: UUID,
        text: str,
        node_type: "_NodeType",
        parents: Optional[Set[UUID]] = None,
        embedding: Optional[np.ndarray] = None,
        score: float = 0.0,
    ) -> "_ThoughtNode":
        node = _ThoughtNode(
            id=node_id,
            text=text,
            node_type=node_type,
            parents=parents or set(),
            embedding=embedding,
            score=score,
        )
        self.nodes[node_id] = node
        
        # Update parent-child relationships
        for parent_id in node.parents:
            if parent_id in self.nodes:
                self.nodes[parent_id].children.add(node_id)
        
        if self.root_id is None:
            self.root_id = node_id
        
        return node
    
    def add_edge(
        self,
        source: UUID,
        target: UUID,
        relation: "_EdgeRelation",
        weight: float = 1.0,
    ) -> "_ThoughtEdge":
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both source and target nodes must exist in the graph")
        
        edge = _ThoughtEdge(
            source=source,
            target=target,
            relation=relation,
            weight=weight,
        )
        self.edges.append(edge)
        
        # Update parent-child relationships
        self.nodes[source].children.add(target)
        self.nodes[target].parents.add(source)
        
        return edge
    
    def get_adjacency_matrix(self) -> np.ndarray:
        n = len(self.nodes)
        if n == 0:
            return np.array([])
        
        node_ids = list(self.nodes.keys())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        A = np.zeros((n, n))
        
        for edge in self.edges:
            if edge.source in node_to_idx and edge.target in node_to_idx:
                i = node_to_idx[edge.source]
                j = node_to_idx[edge.target]
                A[i, j] = edge.weight
        
        return A
    
    def get_node_embeddings_matrix(self) -> np.ndarray:
        node_ids = list(self.nodes.keys())
        if not node_ids:
            return np.array([])
        
        # Get embedding dimension from first node with embedding
        dim = None
        for node_id in node_ids:
            node = self.nodes[node_id]
            if node.embedding is not None:
                dim = len(node.embedding)
                break
        
        if dim is None:
            # Default dimension if no embeddings exist
            dim = 128
        
        H = np.zeros((len(node_ids), dim))
        
        for idx, node_id in enumerate(node_ids):
            node = self.nodes[node_id]
            if node.embedding is not None:
                H[idx] = node.embedding
            else:
                # Zero vector if no embedding
                H[idx] = np.zeros(dim)
        
        return H
    
    def topological_order(self) -> List[UUID]:
        in_degree = defaultdict(int)
        for node_id in self.nodes:
            in_degree[node_id] = len(self.nodes[node_id].parents)
        
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for child_id in self.nodes[node_id].children:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)
        
        # If there are remaining nodes, there's a cycle
        if len(result) < len(self.nodes):
            logger.warning("Graph contains cycles, topological order may be incomplete")
            # Add remaining nodes
            for node_id in self.nodes:
                if node_id not in result:
                    result.append(node_id)
        
        return result


@dataclass
class _GoTConfig:
    
    max_nodes: int = 50
    max_iterations: int = 20
    expansion_branch_factor: int = 3
    merge_similarity_threshold: float = 0.85
    evaluation_temperature: float = 0.3
    expansion_temperature: float = 0.7
    refinement_temperature: float = 0.5
    enable_merging: bool = True
    enable_refinement: bool = True
    enable_feedback: bool = True
    embedding_dim: int = 128
    gnn_layers: int = 2
    gnn_hidden_dim: int = 256
    system_prompt: str = "You are an expert problem solver using graph-based reasoning."
    answer_prefix: str = "Final answer:"
    return_graph: bool = False


class _LLMBackend:
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        raise NotImplementedError("Subclass must implement generate method")


class _GraphEncoder:
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def encode(
        self,
        graph: "_ThoughtGraph",
    ) -> np.ndarray:
        if len(graph.nodes) == 0:
            return np.zeros(self.embedding_dim)
        
        # Get node embeddings matrix H ∈ R^{n×d}
        H = graph.get_node_embeddings_matrix()
        n, d = H.shape
        
        # Get adjacency matrix A ∈ R^{n×n}
        A = graph.get_adjacency_matrix()
        
        # Initialize with node embeddings
        H_current = H.copy()
        
        # Use GraphNeuralNetwork utilities for message passing
        # Full GCN implementation: H^(k+1) = σ(Ã H^(k) W^(k) + b)
        for layer in range(self.num_layers):
            # Use full GCN layer with proper initialization and normalization
            H_next = GraphNeuralNetwork.gcn_layer(
                H_current, A,
                weight_matrix=None,  # Will use Xavier initialization
                bias=None,
                activation="relu",
                dropout=0.0,
                use_layer_norm=(layer > 0)  # Layer norm after first layer
            )
            
            # Residual connection: H^(k+1) = H^(k+1) + H^(k)
            if H_next.shape == H_current.shape:
                H_current = H_next + H_current
            else:
                H_current = H_next
        
        # READOUT: Aggregate node representations using graph readout
        # h_G = Mean/Max/Sum pooling or attention-based pooling
        h_G = GraphNeuralNetwork.graph_readout(H_current, method="mean")
        
        return h_G
    
    def compute_node_embeddings(
        self,
        graph: "_ThoughtGraph",
        input_embedding: Optional[np.ndarray] = None,
    ) -> None:
        # Simple placeholder: use text length and node type as features
        # In practice, replace with actual encoder
        for node_id, node in graph.nodes.items():
            if node.embedding is None:
                # Placeholder embedding based on text and type
                embedding = np.random.randn(self.embedding_dim) * 0.1
                # Add some structure based on text length
                text_len = len(node.text)
                embedding[0] = min(text_len / 100.0, 1.0)
                # Add structure based on node type
                type_idx = hash(node.node_type.value) % self.embedding_dim
                embedding[type_idx % self.embedding_dim] = 1.0
                node.embedding = embedding


class _NodeExpander:
    def __init__(
        self,
        llm: "_LLMBackend",
        config: "_GoTConfig",
    ):
        self.llm = llm
        self.config = config
    
    def expand(
        self,
        graph: "_ThoughtGraph",
        node_id: UUID,
        problem: str,
    ) -> List[str]:
        node = graph.nodes.get(node_id)
        if node is None:
            logger.error(f"Node {node_id} not found in graph")
            return []
        
        # Build context from parent nodes
        parent_texts = []
        for parent_id in node.parents:
            parent_node = graph.nodes.get(parent_id)
            if parent_node:
                parent_texts.append(parent_node.text)
        
        context = "\n".join(parent_texts) if parent_texts else ""
        
        prompt = f"""{self.config.system_prompt}

Problem: {problem}

Current reasoning node:
{node.text}

{("Parent context:\n" + context + "\n") if context else ""}

Propose {self.config.expansion_branch_factor} non-redundant next thoughts that:
1. Refine or extend the current reasoning
2. Break down the problem into subproblems
3. Explore different approaches or perspectives

Return them as numbered items:
1. ...
2. ...
3. ...
..."""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=500 * self.config.expansion_branch_factor,
                temperature=self.config.expansion_temperature,
                top_p=0.9,
                stop=None,
            )
            
            thoughts = self._parse_thoughts(response)
            return thoughts[:self.config.expansion_branch_factor]
        
        except Exception as e:
            logger.error(f"Error expanding node: {e}")
            return []
    
    def _parse_thoughts(self, response: str) -> List[str]:
        thoughts = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove leading number/bullet
            for prefix in [f'{i}.' for i in range(1, 11)] + \
                         [f'{i})' for i in range(1, 11)] + \
                         ['-', '*']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            
            if line:
                thoughts.append(line)
        
        # If no numbered format found, split by newlines
        if not thoughts:
            thoughts = [line.strip() for line in lines if line.strip()]
        
        return thoughts


class _NodeMerger:
    def __init__(
        self,
        llm: "_LLMBackend",
        config: "_GoTConfig",
    ):
        self.llm = llm
        self.config = config
    
    def find_similar_pairs(
        self,
        graph: "_ThoughtGraph",
    ) -> List[Tuple[UUID, UUID, float]]:
        similar_pairs = []
        node_ids = list(graph.nodes.keys())
        
        for i, node_id1 in enumerate(node_ids):
            node1 = graph.nodes[node_id1]
            if node1.embedding is None:
                continue
            
            for node_id2 in node_ids[i+1:]:
                node2 = graph.nodes[node_id2]
                if node2.embedding is None:
                    continue
                
                # Compute cosine similarity
                similarity = self._cosine_similarity(
                    node1.embedding,
                    node2.embedding,
                )
                
                if similarity >= self.config.merge_similarity_threshold:
                    similar_pairs.append((node_id1, node_id2, similarity))
        
        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs
    
    def merge(
        self,
        graph: "_ThoughtGraph",
        node_id1: UUID,
        node_id2: UUID,
        problem: str,
    ) -> Optional[UUID]:
        node1 = graph.nodes.get(node_id1)
        node2 = graph.nodes.get(node_id2)
        
        if node1 is None or node2 is None:
            logger.error("Cannot merge: one or both nodes not found")
            return None
        
        # Synthesize merged text using LLM
        prompt = f"""{self.config.system_prompt}

Problem: {problem}

Thought 1:
{node1.text}

Thought 2:
{node2.text}

Synthesize these two thoughts into a single, coherent thought that combines
the best insights from both while eliminating redundancy.

Merged thought:"""
        
        try:
            merged_text = self.llm.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.5,
                top_p=0.9,
                stop=None,
            ).strip()
            
            # Create merged node
            merged_id = uuid4()
            
            # Union of parents and children
            merged_parents = (node1.parents | node2.parents) - {node_id1, node_id2}
            merged_children = (node1.children | node2.children) - {node_id1, node_id2}
            
            # Average embeddings
            if node1.embedding is not None and node2.embedding is not None:
                merged_embedding = (node1.embedding + node2.embedding) / 2.0
            elif node1.embedding is not None:
                merged_embedding = node1.embedding.copy()
            elif node2.embedding is not None:
                merged_embedding = node2.embedding.copy()
            else:
                merged_embedding = None
            
            # Average scores
            merged_score = (node1.score + node2.score) / 2.0
            
            # Determine node type (prefer more specific types)
            merged_type = node1.node_type
            if node2.node_type in [_NodeType.RESULT, _NodeType.FINAL]:
                merged_type = node2.node_type
            
            merged_node = graph.add_node(
                node_id=merged_id,
                text=merged_text,
                node_type=merged_type,
                parents=merged_parents,
                embedding=merged_embedding,
                score=merged_score,
            )
            
            # Update children to point to merged node
            for child_id in merged_children:
                child = graph.nodes.get(child_id)
                if child:
                    child.parents.discard(node_id1)
                    child.parents.discard(node_id2)
                    child.parents.add(merged_id)
                    merged_node.children.add(child_id)
            
            # Remove old nodes
            del graph.nodes[node_id1]
            del graph.nodes[node_id2]
            
            # Remove edges involving old nodes
            graph.edges = [
                edge for edge in graph.edges
                if edge.source not in [node_id1, node_id2] and
                   edge.target not in [node_id1, node_id2]
            ]
            
            # Add edges from merged parents to merged node
            for parent_id in merged_parents:
                graph.add_edge(parent_id, merged_id, _EdgeRelation.REFINES)
            
            logger.info(f"Merged nodes {node_id1} and {node_id2} into {merged_id}")
            return merged_id
        
        except Exception as e:
            logger.error(f"Error merging nodes: {e}")
            return None
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Normalize to [0, 1] (assuming embeddings are normalized)
        return (similarity + 1.0) / 2.0


class _NodeRefiner:
    def __init__(
        self,
        llm: "_LLMBackend",
        config: "_GoTConfig",
    ):
        self.llm = llm
        self.config = config
    
    def refine(
        self,
        graph: "_ThoughtGraph",
        node_id: UUID,
        problem: str,
    ) -> bool:
        node = graph.nodes.get(node_id)
        if node is None:
            logger.error(f"Node {node_id} not found in graph")
            return False
        
        # Build context from parents and children
        parent_texts = []
        for parent_id in node.parents:
            parent_node = graph.nodes.get(parent_id)
            if parent_node:
                parent_texts.append(parent_node.text)
        
        child_texts = []
        for child_id in node.children:
            child_node = graph.nodes.get(child_id)
            if child_node:
                child_texts.append(child_node.text)
        
        context = ""
        if parent_texts:
            context += "Parent thoughts:\n" + "\n".join(parent_texts) + "\n\n"
        if child_texts:
            context += "Child thoughts:\n" + "\n".join(child_texts) + "\n\n"
        
        prompt = f"""{self.config.system_prompt}

Problem: {problem}

{context}Current reasoning step to improve:
{node.text}

Improve this reasoning step while:
1. Keeping it consistent with parent and child thoughts
2. Making it more precise and clear
3. Ensuring it contributes meaningfully to solving the problem

Improved reasoning step:"""
        
        try:
            refined_text = self.llm.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=self.config.refinement_temperature,
                top_p=0.9,
                stop=None,
            ).strip()
            
            node.text = refined_text
            logger.info(f"Refined node {node_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error refining node: {e}")
            return False


class _NodeEvaluator:
    def __init__(
        self,
        llm: "_LLMBackend",
        config: "_GoTConfig",
    ):
        self.llm = llm
        self.config = config
    
    def evaluate(
        self,
        graph: "_ThoughtGraph",
        node_id: UUID,
        problem: str,
    ) -> float:
        node = graph.nodes.get(node_id)
        if node is None:
            return 0.0
        
        # Build context from ancestors
        ancestor_ids = node.get_ancestors(graph)
        ancestor_texts = []
        for ancestor_id in list(ancestor_ids)[:5]:  # Limit context
            ancestor_node = graph.nodes.get(ancestor_id)
            if ancestor_node:
                ancestor_texts.append(ancestor_node.text)
        
        context = "\n".join(ancestor_texts) if ancestor_texts else ""
        
        prompt = f"""Given this reasoning step in the context of the problem, rate its quality from 0 to 10.

Problem: {problem}

{("Context:\n" + context + "\n") if context else ""}Reasoning step:
{node.text}

Provide a single number from 0 to 10, where:
- 0-3: Very poor, unlikely to help
- 4-6: Moderate quality, somewhat useful
- 7-8: Good quality, likely helpful
- 9-10: Excellent, very promising

Score:"""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=self.config.evaluation_temperature,
                top_p=0.9,
                stop=None,
            )
            
            score = self._parse_score(response)
            node.score = score
            return score
        
        except Exception as e:
            logger.error(f"Error evaluating node: {e}")
            return 0.5  # Default neutral score
    
    def _parse_score(self, response: str) -> float:
        import re
        
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if numbers:
            score = float(numbers[0])
            # Normalize from [0, 10] to [0, 1]
            score = max(0.0, min(10.0, score)) / 10.0
            return score
        
        return 0.5  # Default


class _GoTController:
    def __init__(
        self,
        llm: "_LLMBackend",
        config: "_GoTConfig",
        encoder: "_GraphEncoder",
        expander: "_NodeExpander",
        merger: Optional["_NodeMerger"],
        refiner: Optional["_NodeRefiner"],
        evaluator: "_NodeEvaluator",
    ):
        self.llm = llm
        self.config = config
        self.encoder = encoder
        self.expander = expander
        self.merger = merger
        self.refiner = refiner
        self.evaluator = evaluator
    
    def select_node(
        self,
        graph: "_ThoughtGraph",
        problem: str,
    ) -> Optional[UUID]:
        if len(graph.nodes) == 0:
            return None
        
        # Get adjacency matrix for centrality calculations
        A = graph.get_adjacency_matrix()
        
        if A.size == 0:
            # Fallback to simple selection
            candidate_nodes = list(graph.nodes.items())
            candidate_nodes.sort(key=lambda x: x[1].score)
            return candidate_nodes[0][0] if candidate_nodes else None
        
        # Strategy: combine multiple factors using MDP value estimation
        # Score = w₁·information_gain + w₂·centrality + w₃·uncertainty
        
        candidate_nodes = [
            (node_id, node)
            for node_id, node in graph.nodes.items()
            if len(node.children) == 0 and node.node_type != _NodeType.FINAL
        ]
        
        if not candidate_nodes:
            candidate_nodes = list(graph.nodes.items())
        
        # Calculate composite scores using graph metrics
        node_ids = list(graph.nodes.keys())
        node_scores_enhanced = []
        
        for node_id, node in candidate_nodes:
            if node_id not in node_ids:
                continue
            
            node_idx = node_ids.index(node_id)
            
            # Information gain estimate (based on score uncertainty)
            uncertainty = 1.0 - node.score  # Higher uncertainty = lower score
            
            # Centrality measures
            degree_cent = GraphTopology.degree_centrality(A, node_idx) if node_idx < A.shape[0] else 0.0
            closeness_cent = GraphTopology.closeness_centrality(A, node_idx) if node_idx < A.shape[0] else 0.0
            
            # MDP value estimate (using node score as reward)
            # V(s) = E[Σ γ^k R(s_k)]
            reward_estimate = node.score
            value_estimate = MDPFormulation.value_function_estimate([reward_estimate], gamma=0.99)
            
            # Composite score: information gain + centrality + value
            composite_score = (
                0.4 * uncertainty +  # Information gain component
                0.3 * (degree_cent + closeness_cent) +  # Centrality component
                0.3 * value_estimate  # MDP value component
            )
            
            node_scores_enhanced.append((node_id, node, composite_score))
        
        # Sort by composite score (descending - higher is better)
        node_scores_enhanced.sort(key=lambda x: x[2], reverse=True)
        
        if node_scores_enhanced:
            return node_scores_enhanced[0][0]
        
        return None
    
    def select_operation(
        self,
        graph: "_ThoughtGraph",
        node_id: UUID,
        problem: str,
    ) -> "_GraphOperation":
        # Check if we should stop
        if len(graph.nodes) >= self.config.max_nodes:
            return _GraphOperation.STOP
        
        # Check if we should merge
        if self.config.enable_merging and self.merger:
            similar_pairs = self.merger.find_similar_pairs(graph)
            if similar_pairs:
                return _GraphOperation.MERGE
        
        # Check if we should refine
        if self.config.enable_refinement and self.refiner:
            node = graph.nodes.get(node_id)
            if node and node.score < 0.6:  # Low quality node
                return _GraphOperation.REFINE
        
        # Default: expand
        return _GraphOperation.EXPAND
    
    def step(
        self,
        graph: "_ThoughtGraph",
        problem: str,
    ) -> "_ThoughtGraph":
        # Select node
        node_id = self.select_node(graph, problem)
        if node_id is None:
            return graph
        
        # Select operation
        operation = self.select_operation(graph, node_id, problem)
        
        if operation == _GraphOperation.STOP:
            logger.info("Controller selected STOP operation")
            return graph
        
        elif operation == _GraphOperation.EXPAND:
            # Expand node
            thoughts = self.expander.expand(graph, node_id, problem)
            
            for thought_text in thoughts:
                if len(graph.nodes) >= self.config.max_nodes:
                    break
                
                child_id = uuid4()
                child_node = graph.add_node(
                    node_id=child_id,
                    text=thought_text,
                    node_type=_NodeType.INTERMEDIATE,
                    parents={node_id},
                )
                
                graph.add_edge(node_id, child_id, _EdgeRelation.REFINES)
                
                # Evaluate new node
                self.evaluator.evaluate(graph, child_id, problem)
            
            logger.info(f"Expanded node {node_id} with {len(thoughts)} children")
        
        elif operation == _GraphOperation.MERGE and self.merger:
            # Merge similar nodes
            similar_pairs = self.merger.find_similar_pairs(graph)
            if similar_pairs:
                node_id1, node_id2, similarity = similar_pairs[0]
                self.merger.merge(graph, node_id1, node_id2, problem)
                logger.info(f"Merged nodes {node_id1} and {node_id2}")
        
        elif operation == _GraphOperation.REFINE and self.refiner:
            # Refine node
            self.refiner.refine(graph, node_id, problem)
            # Re-evaluate after refinement
            self.evaluator.evaluate(graph, node_id, problem)
            logger.info(f"Refined node {node_id}")
        
        # Update embeddings for new/modified nodes
        self.encoder.compute_node_embeddings(graph)
        
        return graph


class _AnswerSynthesizer:
    def __init__(
        self,
        llm: "_LLMBackend",
        config: "_GoTConfig",
    ):
        self.llm = llm
        self.config = config
    
    def synthesize(
        self,
        graph: "_ThoughtGraph",
        problem: str,
    ) -> str:
        # Select key nodes (highest scoring, or final nodes)
        key_nodes = []
        
        # First, look for nodes marked as FINAL
        for node_id, node in graph.nodes.items():
            if node.node_type == _NodeType.FINAL:
                key_nodes.append(node)
        
        # If no final nodes, select top-scoring nodes
        if not key_nodes:
            sorted_nodes = sorted(
                graph.nodes.values(),
                key=lambda n: n.score,
                reverse=True,
            )
            key_nodes = sorted_nodes[:min(5, len(sorted_nodes))]
        
        # Build summary of key nodes
        key_texts = [node.text for node in key_nodes]
        summary = "\n\n".join([f"Thought {i+1}: {text}" for i, text in enumerate(key_texts)])
        
        prompt = f"""{self.config.system_prompt}

Problem: {problem}

Key reasoning steps from the graph:
{summary}

Based on the reasoning graph above, provide the final answer to the problem.
Format: {self.config.answer_prefix} [your answer]"""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
                top_p=0.9,
                stop=None,
            )
            
            # Extract answer if prefix is present
            if self.config.answer_prefix in response:
                answer = response.split(self.config.answer_prefix, 1)[1].strip()
            else:
                answer = response.strip()
            
            return answer
        
        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return "Error generating answer"


class GoTAgent:
    def __init__(
        self,
        agent_name: str = "got-agent",
        description: Optional[str] = None,
        model_name: Optional[str] = "gpt-4o",
        llm: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        global_system_prompt: Optional[str] = None,
        secondary_system_prompt: Optional[str] = None,
        config: Optional["_GoTConfig"] = None,
        agent: Optional[Any] = None,
        **kwargs,
    ):
        self.agent_name = agent_name
        self.model_name = model_name
        self.config = config or _GoTConfig()
        
        # Priority: agent > llm > model_name
        if agent is not None:
            # Use provided Agent instance
            self.agent = agent
            llm_adapter = _AgentLLMAdapter(agent)
        elif llm is not None:
            # Use provided LLM directly (can be callable or LLM instance)
            self.agent = llm
            llm_adapter = _AgentLLMAdapter(llm)
        else:
            # Create Agent from model_name
            if model_name is None:
                raise ValueError("Either 'agent', 'llm', or 'model_name' must be provided")
            
            # Import Agent here to avoid circular imports
            from swarms.structs.agent import Agent
            
            # Prepare agent kwargs
            agent_kwargs = {
                "agent_name": agent_name,
                "model_name": model_name,
                **kwargs,
            }
            
            # Add optional parameters if provided
            if description is not None:
                agent_kwargs["agent_description"] = description
            if system_prompt is not None:
                agent_kwargs["system_prompt"] = system_prompt
            if global_system_prompt is not None:
                agent_kwargs["global_system_prompt"] = global_system_prompt
            if secondary_system_prompt is not None:
                agent_kwargs["secondary_system_prompt"] = secondary_system_prompt
            
            self.agent = Agent(**agent_kwargs)
            llm_adapter = _AgentLLMAdapter(self.agent)
        
        # Store the LLM backend for internal use
        self.llm = llm_adapter
        
        # Initialize components
        self.encoder = _GraphEncoder(
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.gnn_hidden_dim,
            num_layers=self.config.gnn_layers,
        )
        
        self.expander = _NodeExpander(self.llm, self.config)
        self.merger = _NodeMerger(self.llm, self.config) if self.config.enable_merging else None
        self.refiner = _NodeRefiner(self.llm, self.config) if self.config.enable_refinement else None
        self.evaluator = _NodeEvaluator(self.llm, self.config)
        self.synthesizer = _AnswerSynthesizer(self.llm, self.config)
        
        self.controller = _GoTController(
            llm=self.llm,
            config=self.config,
            encoder=self.encoder,
            expander=self.expander,
            merger=self.merger,
            refiner=self.refiner,
            evaluator=self.evaluator,
        )
    
    def step(self, task: str, *args, **kwargs) -> str:
        result = self.run(task, return_reasoning=False, *args, **kwargs)
        return result if isinstance(result, str) else result.get("answer", "")
    
    def __getattr__(self, name: str):
        if hasattr(self, 'agent'):
            return getattr(self.agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def run(
        self,
        problem: str,
        return_graph: Optional[bool] = None,
    ) -> Union[str, Dict[str, Any]]:
        return_graph = return_graph if return_graph is not None else self.config.return_graph
        
        logger.info(f"Starting GoT reasoning for problem: {problem[:100]}...")
        
        # 1. Graph initialization
        graph = _ThoughtGraph()
        root_id = uuid4()
        root_node = graph.add_node(
            node_id=root_id,
            text=problem,
            node_type=_NodeType.PROBLEM,
            score=1.0,
        )
        graph.root_id = root_id
        
        # Compute initial embedding
        self.encoder.compute_node_embeddings(graph)
        self.evaluator.evaluate(graph, root_id, problem)
        
        # 2. Iterative graph construction
        for iteration in range(self.config.max_iterations):
            if len(graph.nodes) >= self.config.max_nodes:
                logger.info(f"Reached max nodes ({self.config.max_nodes})")
                break
            
            logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}, nodes: {len(graph.nodes)}")
            
            # Controller step
            graph = self.controller.step(graph, problem)
            
            # Check if controller stopped
            if len(graph.nodes) >= self.config.max_nodes:
                break
        
        # 3. Graph encoding (for potential future use)
        graph_embedding = self.encoder.encode(graph)
        logger.debug(f"Graph embedding computed: shape {graph_embedding.shape}")
        
        # 4. Calculate comprehensive mathematical metrics
        metrics = {}
        
        # Graph topology metrics
        if len(graph.nodes) > 0:
            A = graph.get_adjacency_matrix()
            if A.size > 0:
                # Spectral properties
                laplacian = SpectralGraphTheory.compute_laplacian(A)
                normalized_laplacian = SpectralGraphTheory.compute_normalized_laplacian(A)
                eigenvalues, eigenvectors = SpectralGraphTheory.compute_spectrum(laplacian)
                
                metrics["spectral"] = {
                    "num_eigenvalues": len(eigenvalues),
                    "smallest_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
                    "largest_eigenvalue": float(eigenvalues[-1]) if len(eigenvalues) > 0 else 0.0,
                }
                
                # Topology metrics
                metrics["topology"] = {
                    "diameter": float(GraphTopology.graph_diameter(A)),
                    "degree_distribution": GraphTopology.degree_distribution(A),
                }
                
                # Node centrality (for top nodes) - full implementation
                node_ids = list(graph.nodes.keys())
                if len(node_ids) > 0:
                    centrality_scores = []
                    eigenvector_cent = GraphTopology.eigenvector_centrality(A, max_iter=100, tol=1e-6)
                    
                    for i, node_id in enumerate(node_ids[:10]):  # Limit to first 10
                        if i < A.shape[0]:
                            centrality = GraphTopology.degree_centrality(A, i)
                            clustering = GraphTopology.clustering_coefficient(A, i)
                            closeness = GraphTopology.closeness_centrality(A, i, normalized=True)
                            betweenness = GraphTopology.betweenness_centrality(A, i, normalized=True)
                            eigenvector = float(eigenvector_cent[i]) if i < len(eigenvector_cent) else 0.0
                            
                            centrality_scores.append({
                                "node_id": str(node_id),
                                "degree_centrality": float(centrality),
                                "clustering_coefficient": float(clustering),
                                "closeness_centrality": float(closeness),
                                "betweenness_centrality": float(betweenness),
                                "eigenvector_centrality": eigenvector,
                            })
                    metrics["node_centrality"] = centrality_scores
        
        # Information-theoretic metrics
        node_scores = [node.score for node in graph.nodes.values()]
        if node_scores:
            # Convert scores to probabilities for entropy calculation
            score_probs = [max(0.0, score) for score in node_scores]
            total_score = sum(score_probs)
            if total_score > 0:
                score_probs = [p / total_score for p in score_probs]
                graph_entropy = GraphInformationTheory.graph_entropy(score_probs)
                metrics["information_theory"] = {
                    "graph_entropy": float(graph_entropy),
                }
        
        # Graph complexity
        total_text_length = sum(len(node.text) for node in graph.nodes.values())
        complexity = GraphInformationTheory.graph_complexity(
            num_nodes=len(graph.nodes),
            num_edges=len(graph.edges),
            total_text_length=total_text_length
        )
        metrics["complexity"] = float(complexity)
        
        # Energy-based metrics
        if node_scores:
            # Use log of scores as log probabilities
            log_probs = [math.log(max(0.001, score)) for score in node_scores]
            energies = [GraphEnergyFunction.calculate_graph_energy(logprob) for logprob in log_probs]
            partition_func = GraphEnergyFunction.graph_partition_function(
                energies, self.config.expansion_temperature
            )
            free_energy = GraphEnergyFunction.graph_free_energy(
                partition_func, self.config.expansion_temperature
            )
            
            metrics["energy"] = {
                "partition_function": float(partition_func),
                "free_energy": float(free_energy),
                "avg_energy": float(np.mean(energies)) if energies else 0.0,
            }
        
        # MDP metrics (if we track rewards)
        # This would require tracking rewards during graph construction
        # For now, we can estimate based on node scores
        if node_scores:
            # Estimate value function from node scores
            rewards = [score for score in node_scores if score > 0]
            if rewards:
                value_estimate = MDPFormulation.value_function_estimate(rewards, gamma=0.99)
                # Also compute Q-function and advantage estimates
                if len(rewards) > 1:
                    q_estimate = MDPFormulation.q_function_estimate(
                        state_value=rewards[0],
                        action_reward=rewards[1] if len(rewards) > 1 else rewards[0],
                        next_state_value=rewards[-1],
                        gamma=0.99
                    )
                    advantage = MDPFormulation.advantage_function(q_estimate, value_estimate)
                    
                    metrics["mdp"] = {
                        "value_estimate": float(value_estimate),
                        "q_estimate": float(q_estimate),
                        "advantage": float(advantage),
                    }
                else:
                    metrics["mdp"] = {
                        "value_estimate": float(value_estimate),
                    }
        
        # Graph matching and similarity (compare with previous graph if available)
        # This would be useful for tracking graph evolution
        if len(graph.nodes) > 0 and A.size > 0:
            # Compute graph kernel with itself (self-similarity)
            graph_kernel_self = GraphMatching.graph_kernel(
                graph_embedding, graph_embedding, kernel_type="rbf", gamma=1.0
            )
            metrics["graph_similarity"] = {
                "self_kernel": float(graph_kernel_self),
            }
        
        # 5. Answer synthesis with quantum measurement support
        # If multiple candidate answers exist, use quantum measurement
        final_nodes = [node for node in graph.nodes.values() if node.node_type == _NodeType.FINAL]
        
        if len(final_nodes) > 1:
            # Multiple final nodes: use quantum measurement
            answers = []
            graph_probs = []
            
            for node in final_nodes:
                # Extract answer from node text
                answer_text = node.text
                if self.config.answer_prefix in answer_text:
                    answer_text = answer_text.split(self.config.answer_prefix, 1)[1].strip()
                answers.append(answer_text)
                graph_probs.append(max(0.0, node.score))
            
            # Normalize probabilities
            total_prob = sum(graph_probs)
            if total_prob > 0:
                graph_probs = [p / total_prob for p in graph_probs]
            else:
                graph_probs = [1.0 / len(final_nodes)] * len(final_nodes)
            
            # Quantum measurement: P(y | x) = |⟨y | ψ_G⟩|²
            answer, confidence = QuantumGraphOperations.quantum_graph_measurement(
                graphs=[graph] * len(final_nodes),  # Same graph, different answer nodes
                answers=answers,
                graph_probs=graph_probs
            )
            
            logger.info(f"Quantum measurement selected answer with confidence {confidence:.2f}")
        else:
            # Single or no final nodes: standard synthesis
            answer = self.synthesizer.synthesize(graph, problem)
            confidence = final_nodes[0].score if final_nodes else 0.5
        
        logger.info(f"GoT reasoning complete. Final answer: {answer[:100]}...")
        logger.debug(f"Graph metrics: {metrics}")
        
        if return_graph:
            return {
                "answer": answer,
                "confidence": confidence if len(final_nodes) > 1 else (final_nodes[0].score if final_nodes else 0.5),
                "graph": graph,
                "graph_embedding": graph_embedding,
                "num_nodes": len(graph.nodes),
                "num_edges": len(graph.edges),
                "metrics": metrics,
            }
        
        return answer


class _AgentLLMAdapter(_LLMBackend):
    def __init__(self, agent: Any):
        self.agent = agent
        # Handle both Agent instances and direct LLM instances
        if hasattr(agent, 'llm'):
            self.llm = agent.llm
        elif hasattr(agent, 'run') or callable(agent):
            # Direct LLM instance or callable
            self.llm = agent
        else:
            self.llm = None
    
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        # Try to get LLM from agent if not directly available
        llm = self.llm
        if llm is None and hasattr(self.agent, 'llm'):
            llm = self.agent.llm
        
        if llm is None:
            # Last resort: use agent's run method
            if hasattr(self.agent, 'run'):
                return str(self.agent.run(prompt))
            raise ValueError("No LLM available in agent or adapter")
        
        try:
            # Try to use the LLM's run method directly
            if hasattr(llm, 'run'):
                # Store original parameters if they exist
                original_temp = getattr(llm, 'temperature', None)
                original_top_p = getattr(llm, 'top_p', None)
                original_max_tokens = getattr(llm, 'max_tokens', None)
                
                # Temporarily set parameters
                if hasattr(llm, 'temperature'):
                    llm.temperature = temperature
                if hasattr(llm, 'top_p'):
                    llm.top_p = top_p
                if hasattr(llm, 'max_tokens'):
                    llm.max_tokens = max_tokens
                
                try:
                    result = llm.run(prompt, stop=stop)
                finally:
                    # Restore original parameters
                    if original_temp is not None and hasattr(llm, 'temperature'):
                        llm.temperature = original_temp
                    if original_top_p is not None and hasattr(llm, 'top_p'):
                        llm.top_p = original_top_p
                    if original_max_tokens is not None and hasattr(llm, 'max_tokens'):
                        llm.max_tokens = original_max_tokens
                
                return result if isinstance(result, str) else str(result)
            
            # Fallback: try calling the LLM directly (callable)
            elif callable(llm):
                return str(llm(prompt))
            
            # Last resort: use agent's run method
            else:
                if hasattr(self.agent, 'run'):
                    return str(self.agent.run(prompt))
                raise ValueError("LLM does not have a 'run' method and is not callable")
        
        except Exception as e:
            logger.error(f"Error in LLM adapter: {e}")
            # Last resort: use agent's run method
            if hasattr(self.agent, 'run'):
                return str(self.agent.run(prompt))
            raise


__all__ = ["GoTAgent"]
