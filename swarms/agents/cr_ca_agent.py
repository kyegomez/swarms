"""
CR-CA Lite: A lightweight Causal Reasoning with Counterfactual Analysis engine.

This is a minimal implementation of the ASTT/CR-CA framework focusing on:
- Core evolution operator E(x)
- Counterfactual scenario generation
- Causal chain identification
- Basic causal graph operations

Dependencies: numpy only (typing, dataclasses, enum are stdlib)
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum


class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONFOUNDING = "confounding"
    MEDIATING = "mediating"
    MODERATING = "moderating"


@dataclass
class CausalNode:
    """Represents a node in the causal graph"""
    name: str
    value: Optional[float] = None
    confidence: float = 1.0
    node_type: str = "variable"


@dataclass
class CausalEdge:
    """Represents an edge in the causal graph"""
    source: str
    target: str
    strength: float = 1.0
    relation_type: CausalRelationType = CausalRelationType.DIRECT
    confidence: float = 1.0


@dataclass
class CounterfactualScenario:
    """Represents a counterfactual scenario"""
    name: str
    interventions: Dict[str, float]
    expected_outcomes: Dict[str, float]
    probability: float = 1.0
    reasoning: str = ""


class CRCAgent:
    """
    CR-CA Lite: Lightweight Causal Reasoning with Counterfactual Analysis engine.
    
    Core components:
    - Evolution operator: E(x) = _predict_outcomes()
    - Counterfactual generation: generate_counterfactual_scenarios()
    - Causal chain identification: identify_causal_chain()
    - State mapping: _standardize_state() / _destandardize_value()
    
    Args:
        variables: Optional list of variable names
        causal_edges: Optional list of (source, target) tuples for initial edges
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        causal_edges: Optional[List[Tuple[str, str]]] = None,
        max_loops: Optional[Union[int, str]] = 1,
    ):
        """
        Initialize CR-CA Lite engine.
        
        Args:
            variables: Optional list of variable names to add to graph
            causal_edges: Optional list of (source, target) tuples for initial edges
        """
        # Pure Python graph representation: {node: {child: strength}}
        self.causal_graph: Dict[str, Dict[str, float]] = {}
        self.causal_graph_reverse: Dict[str, List[str]] = {}  # For fast parent lookup
        
        # Standardization statistics: {'var': {'mean': m, 'std': s}}
        self.standardization_stats: Dict[str, Dict[str, float]] = {}
        
        # Initialize graph
        if variables:
            for var in variables:
                self._ensure_node_exists(var)

        if causal_edges:
            for source, target in causal_edges:
                self.add_causal_relationship(source, target)

        # Agent-like loop control: accept numeric or "auto"
        # Keep the original (possibly "auto") value; resolution happens at run time.
        self.max_loops = max_loops

    def _ensure_node_exists(self, node: str) -> None:
        """Ensure node present in graph structures."""
        if node not in self.causal_graph:
            self.causal_graph[node] = {}
        if node not in self.causal_graph_reverse:
            self.causal_graph_reverse[node] = []

    def add_causal_relationship(
        self, 
        source: str, 
        target: str, 
        strength: float = 1.0,
        relation_type: CausalRelationType = CausalRelationType.DIRECT,
        confidence: float = 1.0
    ) -> None:
        """
        Add a causal edge to the graph.
        
        Args:
            source: Source variable name
            target: Target variable name
            strength: Causal effect strength (default: 1.0)
            relation_type: Type of causal relation (default: DIRECT)
            confidence: Confidence in the relationship (default: 1.0)
        """
        # Ensure nodes exist
        self._ensure_node_exists(source)
        self._ensure_node_exists(target)

        # Add or update edge: source -> target with strength
        self.causal_graph[source][target] = float(strength)

        # Update reverse mapping for parent lookup (avoid duplicates)
        if source not in self.causal_graph_reverse[target]:
            self.causal_graph_reverse[target].append(source)
    
    def _get_parents(self, node: str) -> List[str]:
        """
        Get parent nodes (predecessors) of a node.
        
        Args:
            node: Node name
        
        Returns:
            List of parent node names
        """
        return self.causal_graph_reverse.get(node, [])
    
    def _get_children(self, node: str) -> List[str]:
        """
        Get child nodes (successors) of a node.
        
        Args:
            node: Node name
        
        Returns:
            List of child node names
        """
        return list(self.causal_graph.get(node, {}).keys())
    
    def _topological_sort(self) -> List[str]:
        """
        Perform topological sort using Kahn's algorithm (pure Python).
        
        Returns:
            List of nodes in topological order
        """
        # Compute in-degrees
        in_degree: Dict[str, int] = {node: 0 for node in self.causal_graph.keys()}
        for node in self.causal_graph:
            for child in self._get_children(node):
                in_degree[child] = in_degree.get(child, 0) + 1
        
        # Initialize queue with nodes having no incoming edges
        queue: List[str] = [node for node, degree in in_degree.items() if degree == 0]
        result: List[str] = []
        
        # Process nodes
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Reduce in-degree of children
            for child in self._get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    def identify_causal_chain(self, start: str, end: str) -> List[str]:
        """
        Find shortest causal path from start to end using BFS (pure Python).
        
        Implements core causal chain identification (Ax2, Ax6).
        
        Args:
            start: Starting variable
            end: Target variable
        
        Returns:
            List of variables forming the causal chain, or empty list if no path exists
        """
        if start not in self.causal_graph or end not in self.causal_graph:
            return []
        
        if start == end:
            return [start]
        
        # BFS to find shortest path
        queue: List[Tuple[str, List[str]]] = [(start, [start])]
        visited: set = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            # Check all children
            for child in self._get_children(current):
                if child == end:
                    return path + [child]
                
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        
        return []  # No path found
    
    # detect_confounders removed in Lite version (advanced inference)
    
    def _has_path(self, start: str, end: str) -> bool:
        """
        Check if a path exists from start to end using DFS.
        
        Args:
            start: Starting node
            end: Target node
        
        Returns:
            True if path exists, False otherwise
        """
        if start == end:
            return True
        
        stack = [start]
        visited = set()
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for child in self._get_children(current):
                if child == end:
                    return True
                if child not in visited:
                    stack.append(child)
        
        return False
    
    # identify_adjustment_set removed in Lite version (advanced inference)
    
    def _standardize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Standardize state values to z-scores.
        
        Args:
            state: Dictionary of variable values
        
        Returns:
            Dictionary of standardized (z-score) values
        """
        z: Dict[str, float] = {}
        for k, v in state.items():
            s = self.standardization_stats.get(k)
            if s and s.get("std", 0.0) > 0:
                z[k] = (v - s["mean"]) / s["std"]
            else:
                z[k] = v
        return z
    
    def _destandardize_value(self, var: str, z_value: float) -> float:
        """
        Convert z-score back to original scale.
        
        Args:
            var: Variable name
            z_value: Standardized (z-score) value
        
        Returns:
            Original scale value
        """
        s = self.standardization_stats.get(var)
        if s and s.get("std", 0.0) > 0:
            return z_value * s["std"] + s["mean"]
        return z_value
    
    def _predict_outcomes(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evolution operator E(x): Predict outcomes given state and interventions.
        
        This is the core CR-CA evolution operator implementing:
        x_{t+1} = E(x_t)
        
        Mathematical foundation:
        - Linear structural causal model: y = Σᵢ βᵢ·xᵢ + ε\n+        - NOTE: This implementation is linear. To model nonlinear dynamics override\n+          `_predict_outcomes` in a subclass with a custom evolution operator.\n*** End Patch
        - Propagates effects through causal graph in topological order
        - Standardizes inputs, computes in z-space, de-standardizes outputs
        
        Args:
            factual_state: Current world state (baseline)
            interventions: Interventions to apply (do-operator)
        
        Returns:
            Dictionary of predicted variable values
        """
        # Merge factual state with interventions
        raw = factual_state.copy()
        raw.update(interventions)
        
        # Standardize to z-scores
        z_state = self._standardize_state(raw)
        z_pred = dict(z_state)
        
        # Process nodes in topological order
        for node in self._topological_sort():
            # If node is intervened on, keep its value
            if node in interventions:
                if node not in z_pred:
                    z_pred[node] = z_state.get(node, 0.0)
                continue
            
            # Get parents
            parents = self._get_parents(node)
            if not parents:
                continue
            
            # Compute linear combination: Σᵢ βᵢ·z_xi
            s = 0.0
            for p in parents:
                pz = z_pred.get(p, z_state.get(p, 0.0))
                strength = self.causal_graph.get(p, {}).get(node, 0.0)
                s += pz * strength
            
            z_pred[node] = s
        
        # De-standardize results
        return {v: self._destandardize_value(v, z) for v, z in z_pred.items()}
    
    def _calculate_scenario_probability(
        self,
        factual_state: Dict[str, float], 
        interventions: Dict[str, float]
    ) -> float:
        """
        Calculate a heuristic probability of a counterfactual scenario.
        
        NOTE: This is a lightweight heuristic proximity measure (Mahalanobis-like)
        and NOT a full statistical estimator — it ignores covariance and should
        be treated as a relative plausibility score for Lite usage.
        
        Args:
            factual_state: Baseline state
            interventions: Intervention values
        
        Returns:
            Heuristic probability value between 0.05 and 0.98
        """
        z_sq = 0.0
        for var, new in interventions.items():
            s = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
            mu, sd = s.get("mean", 0.0), s.get("std", 1.0) or 1.0
            old = factual_state.get(var, mu)
            dz = (new - mu) / sd - (old - mu) / sd
            z_sq += float(dz) * float(dz)
        
        p = 0.95 * float(np.exp(-0.5 * z_sq)) + 0.05
        return float(max(0.05, min(0.98, p)))
    
    def generate_counterfactual_scenarios(
        self,
        factual_state: Dict[str, float],
        target_variables: List[str],
        max_scenarios: int = 5
    ) -> List[CounterfactualScenario]:
        """
        Generate counterfactual scenarios for target variables.
        
        Implements Ax8 (Counterfactuals) - core CR-CA functionality.
        
        Args:
            factual_state: Current factual state
            target_variables: Variables to generate counterfactuals for
            max_scenarios: Maximum number of scenarios per variable
        
        Returns:
            List of CounterfactualScenario objects
        """
        # Ensure stats exist for variables in factual_state (fallback behavior)
        self.ensure_standardization_stats(factual_state)

        scenarios: List[CounterfactualScenario] = []
        z_steps = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

        for i, tv in enumerate(target_variables[:max_scenarios]):
            stats = self.standardization_stats.get(tv, {"mean": 0.0, "std": 1.0})
            cur = factual_state.get(tv, stats.get("mean", 0.0))

            # If std is zero or missing, use absolute perturbations instead
            if not stats or stats.get("std", 0.0) <= 0:
                base = cur
                abs_steps = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
                vals = [base + step for step in abs_steps]
            else:
                mean = stats["mean"]
                std = stats["std"]
                cz = (cur - mean) / std
                vals = [(cz + dz) * std + mean for dz in z_steps]

            for j, v in enumerate(vals):
                interventions = {tv: float(v)}
                scenarios.append(
                    CounterfactualScenario(
                        name=f"scenario_{i}_{j}",
                        interventions=interventions,
                        expected_outcomes=self._predict_outcomes(
                            factual_state, interventions
                        ),
                        probability=self._calculate_scenario_probability(
                            factual_state, interventions
                        ),
                        reasoning=f"Intervention on {tv} with value {v}",
                    )
                )

        return scenarios
    
    def analyze_causal_strength(self, source: str, target: str) -> Dict[str, float]:
        """
        Analyze the strength of causal relationship between two variables.
        
        Args:
            source: Source variable
            target: Target variable
        
        Returns:
            Dictionary with strength, confidence, path_length, relation_type
        """
        if source not in self.causal_graph or target not in self.causal_graph[source]:
            return {"strength": 0.0, "confidence": 0.0, "path_length": float('inf')}
        
        strength = self.causal_graph[source].get(target, 0.0)
        path = self.identify_causal_chain(source, target)
        path_length = len(path) - 1 if path else float('inf')
        
        return {
            "strength": float(strength),
            "confidence": 1.0,  # Simplified: assume full confidence
            "path_length": path_length,
            "relation_type": CausalRelationType.DIRECT.value
        }
    
    def set_standardization_stats(
        self,
        variable: str,
        mean: float,
        std: float
    ) -> None:
        """
        Set standardization statistics for a variable.
        
        Args:
            variable: Variable name
            mean: Mean value
            std: Standard deviation
        """
        self.standardization_stats[variable] = {"mean": mean, "std": std if std > 0 else 1.0}
    
    def ensure_standardization_stats(self, state: Dict[str, float]) -> None:
        """
        Ensure standardization stats exist for all variables in a given state.
        If stats are missing, create a sensible fallback (mean=observed, std=1.0).
        This prevents degenerate std=0 issues in Lite mode.
        """
        for var, val in state.items():
            if var not in self.standardization_stats:
                self.standardization_stats[var] = {"mean": float(val), "std": 1.0}
    
    def get_nodes(self) -> List[str]:
        """
        Get all nodes in the causal graph.
        
        Returns:
            List of node names
        """
        return list(self.causal_graph.keys())
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """
        Get all edges in the causal graph.
        
        Returns:
            List of (source, target) tuples
        """
        edges = []
        for source, targets in self.causal_graph.items():
            for target in targets.keys():
                edges.append((source, target))
        return edges
    
    def is_dag(self) -> bool:
        """
        Check if the causal graph is a DAG (no cycles).
        
        Uses DFS to detect cycles.
        
        Returns:
            True if DAG, False if cycles exist
        """
        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            """DFS to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            
            for child in self._get_children(node):
                if child not in visited:
                    if has_cycle(child, visited, rec_stack):
                        return True
                elif child in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        rec_stack = set()
        
        for node in self.causal_graph:
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    return False
        
        return True
    
    def run(
        self,
        initial_state: Any,
        target_variables: Optional[List[str]] = None,
        max_steps: Union[int, str] = 1
    ) -> Dict[str, Any]:
        """
        Run causal simulation: evolve state and generate counterfactuals.
        
        Simple entry point for CR-CA engine.
        
        Args:
            initial_state: Initial world state
            target_variables: Variables to generate counterfactuals for (default: all nodes)
            max_steps: Number of evolution steps (default: 1)
        
        Returns:
            Dictionary with evolved state, counterfactuals, and graph info
        """
        # Accept either a dict initial_state or a JSON string (agent-like behavior)
        if not isinstance(initial_state, dict):
            try:
                import json
                parsed = json.loads(initial_state)
                if isinstance(parsed, dict):
                    initial_state = parsed
                else:
                    return {"error": "initial_state JSON must decode to a dict"}
            except Exception:
                return {"error": "initial_state must be a dict or JSON-encoded dict"}

        # Use all nodes as targets if not specified
        if target_variables is None:
            target_variables = list(self.causal_graph.keys())
        
        # Resolve "auto" sentinel for max_steps (accepts method arg or instance-level default)
        def _resolve_max_steps(value: Union[int, str]) -> int:
            if isinstance(value, str) and value == "auto":
                # Heuristic: one step per variable (at least 1)
                return max(1, len(self.causal_graph))
            try:
                return int(value)
            except Exception:
                return max(1, len(self.causal_graph))

        effective_steps = _resolve_max_steps(max_steps if max_steps != 1 or self.max_loops == 1 else self.max_loops)
        # If caller passed default 1 and instance set a different max_loops, prefer instance value
        if max_steps == 1 and self.max_loops != 1:
            effective_steps = _resolve_max_steps(self.max_loops)

        # Evolve state
        current_state = initial_state.copy()
        for step in range(effective_steps):
            current_state = self._predict_outcomes(current_state, {})
        
        # Ensure standardization stats exist for the evolved state and generate counterfactuals from it
        self.ensure_standardization_stats(current_state)
        counterfactual_scenarios = self.generate_counterfactual_scenarios(
            current_state,
            target_variables,
            max_scenarios=5
        )
        
        return {
            "initial_state": initial_state,
            "evolved_state": current_state,
            "counterfactual_scenarios": counterfactual_scenarios,
            "causal_graph_info": {
                "nodes": self.get_nodes(),
                "edges": self.get_edges(),
                "is_dag": self.is_dag()
            },
            "steps": effective_steps
        }


# Agent-like behavior: `run` accepts either a dict or a JSON string as the initial_state
# so the engine behaves like a normal agent by default.
