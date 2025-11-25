"""
CR-CA Lite: A lightweight Causal Reasoning with Counterfactual Analysis engine.

This is a minimal implementation of the ASTT/CR-CA framework focusing on:
- Core evolution operator E(x)
- Counterfactual scenario generation
- Causal chain identification
- Basic causal graph operations

Dependencies: numpy only (typing, dataclasses, enum are stdlib)
"""

from typing import Dict, Any, List, Tuple, Optional
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


class CRCALite:
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
                if var not in self.causal_graph:
                    self.causal_graph[var] = {}
                if var not in self.causal_graph_reverse:
                    self.causal_graph_reverse[var] = []
        
        if causal_edges:
            for source, target in causal_edges:
                self.add_causal_relationship(source, target)

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
        # Initialize nodes if needed
        if source not in self.causal_graph:
            self.causal_graph[source] = {}
        if target not in self.causal_graph:
            self.causal_graph[target] = {}
        if source not in self.causal_graph_reverse:
            self.causal_graph_reverse[source] = []
        if target not in self.causal_graph_reverse:
            self.causal_graph_reverse[target] = []
        
        # Add edge: source -> target with strength
        self.causal_graph[source][target] = strength
        
        # Update reverse mapping for parent lookup
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
    
    def detect_confounders(self, treatment: str, outcome: str) -> List[str]:
        """
        Detect confounders: variables that are ancestors of both treatment and outcome.
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
        
        Returns:
            List of confounder variable names
        """
        def get_ancestors(node: str) -> set:
            """Get all ancestors of a node using DFS."""
            ancestors = set()
            stack = [node]
            visited = set()
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                
                parents = self._get_parents(current)
                for parent in parents:
                    if parent not in ancestors:
                        ancestors.add(parent)
                        stack.append(parent)
            
            return ancestors
        
        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []
        
        t_ancestors = get_ancestors(treatment)
        o_ancestors = get_ancestors(outcome)
        
        # Confounders are common ancestors
        confounders = list(t_ancestors & o_ancestors)
        
        # Verify they have paths to both treatment and outcome
        valid_confounders = []
        for conf in confounders:
            if (self._has_path(conf, treatment) and self._has_path(conf, outcome)):
                valid_confounders.append(conf)
        
        return valid_confounders
    
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
    
    def identify_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """
        Identify back-door adjustment set for causal effect estimation.
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
        
        Returns:
            List of variables in the adjustment set
        """
        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []
        
        # Get parents of treatment
        parents_t = set(self._get_parents(treatment))
        
        # Get descendants of treatment
        def get_descendants(node: str) -> set:
            """Get all descendants using DFS."""
            descendants = set()
            stack = [node]
            visited = set()
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                
                for child in self._get_children(current):
                    if child not in descendants:
                        descendants.add(child)
                        stack.append(child)
            
            return descendants
        
        descendants_t = get_descendants(treatment)
        
        # Adjustment set: parents of treatment that are not descendants and not the outcome
        adjustment = [
            z for z in parents_t 
            if z not in descendants_t and z != outcome
        ]
        
        return adjustment
    
    def _standardize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Standardize state values to z-scores.
        
        Args:
            state: Dictionary of variable values
        
        Returns:
            Dictionary of standardized (z-score) values
        """
        z = {}
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
        - Linear structural causal model: y = Σᵢ βᵢ·xᵢ + ε
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
        Calculate probability of a counterfactual scenario.
        
        Uses Mahalanobis distance in standardized space.
        
        Args:
            factual_state: Baseline state
            interventions: Intervention values
        
        Returns:
            Probability value between 0.05 and 0.98
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
        scenarios = []
        z_steps = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        
        for i, tv in enumerate(target_variables[:max_scenarios]):
            s = self.standardization_stats.get(tv, {"mean": 0.0, "std": 1.0})
            cur = factual_state.get(tv, 0.0)
            cz = (cur - s["mean"]) / s["std"] if s["std"] > 0 else 0.0
            
            vals = [(cz + dz) * s["std"] + s["mean"] for dz in z_steps]
            
            for j, v in enumerate(vals):
                scenarios.append(CounterfactualScenario(
                    name=f"scenario_{i}_{j}",
                    interventions={tv: v},
                    expected_outcomes=self._predict_outcomes(factual_state, {tv: v}),
                    probability=self._calculate_scenario_probability(factual_state, {tv: v}),
                    reasoning=f"Intervention on {tv} with value {v}"
                ))
        
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
        initial_state: Dict[str, float],
        target_variables: Optional[List[str]] = None,
        max_steps: int = 1
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
        # Use all nodes as targets if not specified
        if target_variables is None:
            target_variables = list(self.causal_graph.keys())
        
        # Evolve state
        current_state = initial_state.copy()
        for step in range(max_steps):
            current_state = self._predict_outcomes(current_state, {})
        
        # Generate counterfactual scenarios
        counterfactual_scenarios = self.generate_counterfactual_scenarios(
            initial_state,
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
            "steps": max_steps
        }
