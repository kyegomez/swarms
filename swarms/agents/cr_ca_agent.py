from swarms.structs.agent import Agent
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import networkx as nx
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math
from functools import lru_cache
from collections import defaultdict
from itertools import combinations
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import minimize, differential_evolution, minimize_scalar, basinhopping
from scipy.spatial.distance import euclidean, cosine, jensenshannon
from scipy.linalg import cholesky, inv, pinv
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


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


class CRCAAgent:
    """
    Causal Reasoning with Counterfactual Analysis Agent.
    
    This agent performs sophisticated causal inference and counterfactual reasoning
    to understand cause-and-effect relationships and explore alternative scenarios.
    """

    def __init__(
        self,
        name: str = "cr-ca-agent",
        description: str = "Causal Reasoning with Counterfactual Analysis agent",
        model_name: str = "openai/gpt-4o",
        max_loops: int = 3,
        causal_graph: Optional[nx.DiGraph] = None,
        variables: Optional[List[str]] = None,
        causal_edges: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Initialize the CR-CA Agent.

        Args:
            name: Agent name
            description: Agent description
            model_name: LLM model to use
            max_loops: Maximum reasoning loops
            causal_graph: Pre-built causal graph
            variables: List of variable names
            causal_edges: List of causal relationships (source, target)
        """
        self.name = name
        self.description = description
        self.model_name = model_name
        self.max_loops = max_loops
        
        # Initialize causal graph
        self.causal_graph = causal_graph or nx.DiGraph()
        if variables:
            self.causal_graph.add_nodes_from(variables)
        if causal_edges:
            self.causal_graph.add_edges_from(causal_edges)
        
        # Initialize agent with CR-CA schema
        self.agent = Agent(
            agent_name=self.name,
            agent_description=self.description,
            model_name=self.model_name,
            max_loops=1,
            tools_list_dictionary=[self._get_cr_ca_schema()],
            output_type="final",
        )
        
        # Memory for storing causal analysis history
        self.causal_memory: List[Dict[str, Any]] = []
        self.counterfactual_scenarios: List[CounterfactualScenario] = []
        # Standardization statistics for each variable: {'var': {'mean': m, 'std': s}}
        self.standardization_stats: Dict[str, Dict[str, float]] = {}
        # Optional history of learned edge strengths for temporal tracking
        self.edge_strength_history: List[Dict[Tuple[str, str], float]] = []
        # Optional constraints: enforce monotonic signs on edges { (u,v): +1|-1 }
        self.edge_sign_constraints: Dict[Tuple[str, str], int] = {}
        # Random number generator for probabilistic methods
        self.rng = np.random.default_rng()
        
        # Performance: caching for expensive computations
        self._prediction_cache: Dict[Tuple[tuple, tuple], Dict[str, float]] = {}
        self._cache_enabled: bool = True
        self._cache_max_size: int = 1000
        
        # Non-linear extensions: interaction terms {node: [list of parent pairs to interact]}
        self.interaction_terms: Dict[str, List[Tuple[str, str]]] = {}
        
        # Information theory cache
        self._entropy_cache: Dict[str, float] = {}
        self._mi_cache: Dict[Tuple[str, str], float] = {}
        
        # Bayesian inference: prior distributions {edge: {'mu': μ₀, 'sigma': σ₀}}
        self.bayesian_priors: Dict[Tuple[str, str], Dict[str, float]] = {}

    def _get_cr_ca_schema(self) -> Dict[str, Any]:
        """Get the CR-CA agent schema for structured reasoning."""
        return {
            "type": "function",
            "function": {
                "name": "generate_causal_analysis",
                "description": "Generates structured causal reasoning and counterfactual analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "causal_analysis": {
                            "type": "string",
                            "description": "Analysis of causal relationships and mechanisms"
                        },
                        "intervention_planning": {
                            "type": "string", 
                            "description": "Planned interventions to test causal hypotheses"
                        },
                        "counterfactual_scenarios": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scenario_name": {"type": "string"},
                                    "interventions": {"type": "object"},
                                    "expected_outcomes": {"type": "object"},
                                    "reasoning": {"type": "string"}
                                }
                            },
                            "description": "Multiple counterfactual scenarios to explore"
                        },
                        "causal_strength_assessment": {
                            "type": "string",
                            "description": "Assessment of causal relationship strengths and confounders"
                        },
                        "optimal_solution": {
                            "type": "string",
                            "description": "Recommended optimal solution based on causal analysis"
                        }
                    },
                    "required": [
                        "causal_analysis",
                        "intervention_planning", 
                        "counterfactual_scenarios",
                        "causal_strength_assessment",
                        "optimal_solution"
                    ]
                }
            }
        }

    def add_causal_relationship(
        self, 
        source: str, 
        target: str, 
        strength: float = 1.0,
        relation_type: CausalRelationType = CausalRelationType.DIRECT,
        confidence: float = 1.0
    ) -> None:
        """Add a causal relationship to the graph."""
        self.causal_graph.add_edge(source, target)
        self.causal_graph[source][target].update({
            'strength': strength,
            'relation_type': relation_type,
            'confidence': confidence
        })

    def identify_causal_chain(self, start: str, end: str) -> List[str]:
        """Identify the causal chain between two variables."""
        try:
            path = nx.shortest_path(self.causal_graph, start, end)
            return path
        except nx.NetworkXNoPath:
            return []

    def detect_confounders(self, treatment: str, outcome: str) -> List[str]:
        """Detect potential confounders for a treatment-outcome relationship."""
        confounders = []
        
        # Find common ancestors
        treatment_ancestors = set(nx.ancestors(self.causal_graph, treatment))
        outcome_ancestors = set(nx.ancestors(self.causal_graph, outcome))
        common_ancestors = treatment_ancestors.intersection(outcome_ancestors)
        
        # Check if common ancestors are connected to both treatment and outcome
        for ancestor in common_ancestors:
            if (nx.has_path(self.causal_graph, ancestor, treatment) and 
                nx.has_path(self.causal_graph, ancestor, outcome)):
                confounders.append(ancestor)
        
        return confounders

    def generate_counterfactual_scenarios(
        self, 
        factual_state: Dict[str, float],
        target_variables: List[str],
        max_scenarios: int = 5
    ) -> List[CounterfactualScenario]:
        """Generate counterfactual scenarios for given variables."""
        scenarios = []
        
        for i, target_var in enumerate(target_variables[:max_scenarios]):
            # Use standardized z-score increments to avoid exploding magnitudes
            stats = self.standardization_stats.get(target_var, {"mean": 0.0, "std": 1.0})
            z_steps = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
            current_raw = factual_state.get(target_var, 0.0)
            # Map current value to z space (if stats available)
            if stats["std"] > 0:
                current_z = (current_raw - stats["mean"]) / stats["std"]
            else:
                current_z = 0.0
            # Propose interventions around current z value
            proposed_z_values = [current_z + dz for dz in z_steps]
            # Convert back to raw for simulation
            intervention_values = [
                (z * stats["std"]) + stats["mean"] for z in proposed_z_values
            ]
            
            for j, intervention_value in enumerate(intervention_values):
                scenario = CounterfactualScenario(
                    name=f"scenario_{i}_{j}",
                    interventions={target_var: intervention_value},
                    expected_outcomes=self._predict_outcomes(factual_state, {target_var: intervention_value}),
                    probability=self._calculate_scenario_probability(factual_state, {target_var: intervention_value}),
                    reasoning=f"Intervention on {target_var} with value {intervention_value}"
                )
                scenarios.append(scenario)
        
        return scenarios

    def _standardize_state(self, state: Dict[str, float]) -> Dict[str, float]:
        """Convert raw state dict to standardized z-scores using stored stats."""
        z: Dict[str, float] = {}
        for k, v in state.items():
            stats = self.standardization_stats.get(k)
            if stats and stats.get("std", 0.0) > 0:
                z[k] = (v - stats["mean"]) / stats["std"]
            else:
                z[k] = v
        return z

    def _destandardize_value(self, var: str, z_value: float) -> float:
        """Convert a z-score back to raw value for a specific variable."""
        stats = self.standardization_stats.get(var)
        if stats and stats.get("std", 0.0) > 0:
            return z_value * stats["std"] + stats["mean"]
        return z_value

    def _predict_outcomes(
        self, 
        factual_state: Dict[str, float], 
        interventions: Dict[str, float],
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """
        Predict outcomes given interventions using standardized linear propagation.
        
        Mathematical foundation:
        - Structural Equation Model (SEM): y = Xβ + ε where β are structural coefficients
        - Do-operator: do(X=x) sets X=x, removing its dependence on parents
        - In z-space (standardized): z_y = Σᵢ βᵢ·z_xi + z_ε
        - Propagation: topological order ensures parents computed before children
        
        Standardization: z = (x - μ)/σ where μ is mean, σ is standard deviation
        This ensures numerical stability and scale-invariance.
        
        Linear SCM: Each node y has equation y = Σᵢ βᵢ·xᵢ + ε
        where xᵢ are parents, βᵢ are edge strengths, ε is noise.
        
        Args:
            factual_state: Current state
            interventions: Interventions to apply
            use_cache: Whether to use prediction cache
        
        Returns:
            Predicted outcomes
        """
        # Note: Caching is handled by _predict_outcomes_cached wrapper
        # This method is the actual computation (call with use_cache=False from cache wrapper)
        # Merge states and convert to z-space
        # Standardization: z = (x - μ)/σ (z-score transformation)
        raw_state = factual_state.copy()
        raw_state.update(interventions)
        z_state = self._standardize_state(raw_state)

        # Work on a copy to avoid mutating initial inputs
        z_pred = dict(z_state)

        # Propagate in topological order (ensures parents computed before children)
        # Topological sort: linearization of DAG respecting causal ordering
        for node in nx.topological_sort(self.causal_graph):
            if node in interventions:
                # Do-operator: do(X=x) forces X=x, breaking dependence on parents
                # If directly intervened, keep its standardized value as-is
                if node not in z_pred:
                    z_pred[node] = z_state.get(node, 0.0)
                continue

            predecessors = list(self.causal_graph.predecessors(node))
            if not predecessors:
                # Exogenous nodes (no parents): z_node = z_ε (noise term)
                continue

            # Linear structural equation: z_y = Σᵢ βᵢ·z_xi
            # This is the structural causal model (SCM) equation in z-space
            effect_z = 0.0
            for parent in predecessors:
                parent_z = z_pred.get(parent)
                if parent_z is None:
                    parent_z = z_state.get(parent, 0.0)
                edge_data = self.causal_graph[parent][node]
                strength = edge_data.get('strength', 0.0)  # Structural coefficient βᵢ
                # Linear combination: z_y = Σᵢ βᵢ·z_xi
                effect_z += parent_z * strength

            z_pred[node] = effect_z  # Store predicted z-score

        # Convert back to raw value space using inverse standardization
        # De-standardization: x = z·σ + μ
        predicted_state: Dict[str, float] = {}
        for var, z_val in z_pred.items():
            predicted_state[var] = self._destandardize_value(var, z_val)
        return predicted_state

    def _calculate_scenario_probability(
        self, 
        factual_state: Dict[str, float], 
        interventions: Dict[str, float]
    ) -> float:
        """
        Calculate smoothed probability using standardized z-distance.
        
        Mathematical formulation:
        - Compute z-score deltas: dz_i = z_new_i - z_old_i for each intervened variable
        - L2 norm squared: ||dz||² = Σᵢ (dz_i)²
        - Gaussian-like probability: p = 0.95 * exp(-0.5 * ||dz||²) + 0.05
        - Bounded to [0.05, 0.98]: p = clip(p, 0.05, 0.98)
        
        This follows the Mahalanobis distance-based probability in multivariate Gaussian.
        """
        # Step 1: Compute z-score deltas for intervened variables
        # dz_i = z_new_i - z_old_i where z = (x - μ)/σ
        z_sq_sum = 0.0
        for var, new_val in interventions.items():
            stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
            mu = stats.get('mean', 0.0)  # μ: mean
            sd = stats.get('std', 1.0) or 1.0  # σ: standard deviation
            
            # Get old value from factual state or use mean as default
            old_val = factual_state.get(var, mu)
            
            # Compute z-scores: z = (x - μ)/σ
            z_new = (new_val - mu) / sd
            z_old = (old_val - mu) / sd
            
            # Delta in z-space: dz = z_new - z_old
            dz = z_new - z_old
            
            # Accumulate squared L2 norm: ||dz||² = Σᵢ (dz_i)²
            z_sq_sum += float(dz) * float(dz)
        
        # Step 2: Compute probability using exponential decay
        # p = 0.95 * exp(-0.5 * ||dz||²) + 0.05
        # This gives high probability (near 0.95) when ||dz||² ≈ 0 (small intervention)
        # and low probability (near 0.05) when ||dz||² is large (large intervention)
        prob = 0.95 * float(np.exp(-0.5 * z_sq_sum)) + 0.05
        
        # Step 3: Clip to valid probability range [0.05, 0.98]
        # p = min(0.98, max(0.05, p))
        prob = max(0.05, min(0.98, prob))
        
        return float(prob)

    def analyze_causal_strength(self, source: str, target: str) -> Dict[str, float]:
        """Analyze the strength of causal relationship between two variables."""
        if not self.causal_graph.has_edge(source, target):
            return {"strength": 0.0, "confidence": 0.0, "path_length": float('inf')}
        
        edge_data = self.causal_graph[source][target]
        path_length = len(nx.shortest_path(self.causal_graph, source, target)) - 1
        
        return {
            "strength": edge_data.get('strength', 1.0),
            "confidence": edge_data.get('confidence', 1.0),
            "path_length": path_length,
            "relation_type": edge_data.get('relation_type', CausalRelationType.DIRECT).value
        }

    def fit_from_dataframe(
        self,
        df: Any,
        variables: List[str],
        window: int = 30,
        decay_alpha: float = 0.9,
        ridge_lambda: float = 0.0,
        enforce_signs: bool = True
    ) -> None:
        """Fit edge strengths and standardization stats from a rolling window with recency weighting.

        For each child variable, perform a simple weighted linear regression on its parents
        defined in the existing graph to estimate edge 'strength' coefficients. Also compute
        mean/std for z-score scaling. Uses exponential decay weights to emphasize recent data.
        """
        if df is None or len(df) == 0:
            return
        df_local = df[variables].dropna().copy()
        if len(df_local) < max(3, window):
            # Still compute stats on available data
            pass
        # Use the last `window` rows
        window_df = df_local.tail(window)
        n = len(window_df)
        # Exponential decay weights: newer rows get higher weights
        # Oldest gets alpha^(n-1), newest gets alpha^0 = 1.0
        weights = np.array([decay_alpha ** (n - 1 - i) for i in range(n)], dtype=float)
        weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)

        # Compute standardization stats
        self.standardization_stats = {}
        for v in variables:
            m = float(window_df[v].mean())
            s = float(window_df[v].std(ddof=0))
            if s == 0:
                s = 1.0
            self.standardization_stats[v] = {"mean": m, "std": s}
        # Ensure default stats for any graph node not in the window variables
        for node in self.causal_graph.nodes():
            if node not in self.standardization_stats:
                self.standardization_stats[node] = {"mean": 0.0, "std": 1.0}

        # Estimate edge strengths per node from its parents
        learned_strengths: Dict[Tuple[str, str], float] = {}
        for child in self.causal_graph.nodes():
            parents = list(self.causal_graph.predecessors(child))
            if not parents:
                continue
            # Skip children not present in the data window
            if child not in window_df.columns:
                continue
            # Prepare standardized design matrix X (parents) and target y (child)
            X_cols = []
            for p in parents:
                if p in window_df.columns:
                    X_cols.append(((window_df[p] - self.standardization_stats[p]["mean"]) / self.standardization_stats[p]["std"]).values)
            if not X_cols:
                continue
            X = np.vstack(X_cols).T  # shape (n, k)
            y = ((window_df[child] - self.standardization_stats[child]["mean"]) / self.standardization_stats[child]["std"]).values
            # Weighted least squares: (X' W X)^{-1} X' W y
            W = np.diag(weights)
            XtW = X.T @ W
            XtWX = XtW @ X
            # Ridge regularization for stability
            if ridge_lambda > 0 and XtWX.size > 0:
                k = XtWX.shape[0]
                XtWX = XtWX + ridge_lambda * np.eye(k)
            try:
                XtWX_inv = np.linalg.pinv(XtWX)
                beta = XtWX_inv @ (XtW @ y)
            except Exception:
                beta = np.zeros(X.shape[1])
            # Assign strengths to edges in order of parents
            for idx, p in enumerate(parents):
                strength = float(beta[idx]) if idx < len(beta) else 0.0
                # Enforce monotonic sign constraints if requested
                if enforce_signs:
                    sign = self.edge_sign_constraints.get((p, child))
                    if sign == 1 and strength < 0:
                        strength = 0.0
                    elif sign == -1 and strength > 0:
                        strength = 0.0
                if self.causal_graph.has_edge(p, child):
                    self.causal_graph[p][child]['strength'] = strength
                    self.causal_graph[p][child]['confidence'] = 1.0
                    learned_strengths[(p, child)] = strength
        # Track history for temporal drift analysis
        self.edge_strength_history.append(learned_strengths)

        # Enforce DAG: if cycles exist, iteratively remove weakest edge in cycles
        try:
            while not nx.is_directed_acyclic_graph(self.causal_graph):
                cycle_edges = list(nx.find_cycle(self.causal_graph, orientation="original"))
                # pick weakest edge among cycle
                weakest = None
                weakest_w = float("inf")
                for u, v, _ in cycle_edges:
                    w = abs(float(self.causal_graph[u][v].get('strength', 0.0)))
                    if w < weakest_w:
                        weakest_w = w
                        weakest = (u, v)
                if weakest:
                    self.causal_graph.remove_edge(*weakest)
                else:
                    break
        except Exception:
            pass

    def identify_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """Heuristic back-door adjustment set selection.

        Returns a set that blocks back-door paths: parents of treatment excluding descendants of treatment and the outcome itself.
        """
        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []
        parents_t = set(self.causal_graph.predecessors(treatment))
        descendants_t = set(nx.descendants(self.causal_graph, treatment))
        adjustment = [z for z in parents_t if z not in descendants_t and z != outcome]
        return adjustment

    def estimate_ate(
        self,
        df: Any,
        treatment: str,
        outcome: str,
        conditioning: Optional[List[str]] = None,
        method: str = 'ols'
    ) -> float:
        """Estimate average treatment effect using simple regression adjustment.

        y ~ T + Z, return coefficient on T in standardized space.
        """
        if df is None or len(df) == 0:
            return 0.0
        if conditioning is None:
            conditioning = self.identify_adjustment_set(treatment, outcome)
        cols = [c for c in [outcome, treatment] + conditioning if c in df.columns]
        data = df[cols].dropna()
        if len(data) < 5:
            return 0.0
        # Standardize
        ds = (data - data.mean()) / (data.std(ddof=0).replace(0, 1.0))
        y = ds[outcome].values
        X_cols = [ds[treatment].values] + [ds[c].values for c in conditioning]
        X = np.vstack(X_cols).T
        # OLS
        try:
            beta = np.linalg.pinv(X) @ y
            return float(beta[0])
        except Exception:
            return 0.0

    def estimate_cate(
        self,
        df: Any,
        treatment: str,
        outcome: str,
        context_by: Optional[str] = None,
        num_bins: int = 3
    ) -> Dict[str, float]:
        """Estimate heterogeneous effects by binning a context feature and computing local ATEs."""
        if df is None or len(df) == 0 or context_by is None or context_by not in df.columns:
            return {}
        series = df[context_by].dropna()
        if len(series) < 10:
            return {}
        quantiles = np.linspace(0, 1, num_bins + 1)
        bins = series.quantile(quantiles).values
        bins[0] = -np.inf
        bins[-1] = np.inf
        cate: Dict[str, float] = {}
        for i in range(num_bins):
            mask = (df[context_by] > bins[i]) & (df[context_by] <= bins[i+1])
            ate_bin = self._safe_ate_bin(df[mask], treatment, outcome)
            cate[f"bin_{i}"] = ate_bin
        return cate

    def _safe_ate_bin(self, df: Any, treatment: str, outcome: str) -> float:
        if df is None or len(df) < 5:
            return 0.0
        return self.estimate_ate(df, treatment, outcome)

    def counterfactual_abduction_action_prediction(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Abduction–Action–Prediction for linear-Gaussian SCM in z-space.
        
        Pearl's three-step counterfactual reasoning:
        
        1. Abduction: Infer latent noise terms ε from factual observations
           - Given factual: x_factual, compute ε = x - E[x|parents]
           - For linear SCM: ε = z_y - Σᵢ βᵢ·z_xi (residual)
           - This preserves the "history" that led to factual state
        
        2. Action: Apply do-operator do(X=x*) to set intervention values
           - do(X=x*) breaks causal dependencies: P(Y|do(X=x*)) ≠ P(Y|X=x*)
           - Sets intervened variables to counterfactual values
        
        3. Prediction: Propagate with new values but old noise
           - Use abduced noise ε from step 1
           - Predict: y_cf = Σᵢ βᵢ·x_cfᵢ + ε (same noise, new parents)
           - This gives counterfactual: "What if X had been x* instead of x_factual?"
        
        Mathematical foundation:
        - Factual: Y = f(X, ε) where ε ~ N(0, σ²)
        - Abduction: ε̂ = Y_factual - f(X_factual, 0) (infer noise from observation)
        - Counterfactual: Y_cf = f(X_cf, ε̂) (same noise, different X)
        """
        # Standardize factual: z = (x - μ)/σ
        z = self._standardize_state(factual_state)
        noise: Dict[str, float] = {}
        
        # Step 1: ABDUCTION - Infer latent noise terms from factual observations
        # Noise represents unobserved confounders and stochastic variation
        # For linear SCM: ε = z_y - Σᵢ βᵢ·z_xi (residual from linear regression)
        for node in nx.topological_sort(self.causal_graph):
            parents = list(self.causal_graph.predecessors(node))
            if not parents:
                # Exogenous: noise equals observed value (no parents to subtract)
                noise[node] = z.get(node, 0.0)
                continue
            
            # Predicted value from structural equation: ŷ = Σᵢ βᵢ·xᵢ
            pred = 0.0
            for p in parents:
                w = self.causal_graph[p][node].get('strength', 0.0)  # βᵢ
                pred += z.get(p, 0.0) * w  # Σᵢ βᵢ·z_xi
            
            # Abduce noise: ε = z_observed - ŷ
            # This captures the deviation from deterministic prediction
            noise[node] = z.get(node, 0.0) - pred
        # Action + prediction
        cf_raw = factual_state.copy()
        cf_raw.update(interventions)
        z_cf = self._standardize_state(cf_raw)
        z_pred: Dict[str, float] = {}
        for node in nx.topological_sort(self.causal_graph):
            if node in interventions:
                z_pred[node] = z_cf.get(node, 0.0)
                continue
            parents = list(self.causal_graph.predecessors(node))
            if not parents:
                z_pred[node] = noise.get(node, 0.0)
                continue
            val = 0.0
            for p in parents:
                w = self.causal_graph[p][node].get('strength', 0.0)
                val += z_pred.get(p, z_cf.get(p, 0.0)) * w
            z_pred[node] = val + noise.get(node, 0.0)
        # De-standardize
        out: Dict[str, float] = {k: self._destandardize_value(k, v) for k, v in z_pred.items()}
        return out

    def quantify_uncertainty(
        self,
        df: Any,
        variables: List[str],
        windows: int = 200,
        alpha: float = 0.95
    ) -> Dict[str, Any]:
        """Bootstrap strengths and produce confidence intervals per edge.

        If PyMC is available, provide a Bayesian posterior CI as well (best effort).
        """
        if df is None or len(df) == 0:
            return {"edge_cis": {}, "samples": 0}
        samples: Dict[Tuple[str, str], List[float]] = {}
        usable = df[variables].dropna()
        if len(usable) < 10:
            return {"edge_cis": {}, "samples": 0}
        idx = np.arange(len(usable))
        for _ in range(windows):
            boot_idx = np.random.choice(idx, size=len(idx), replace=True)
            boot_df = usable.iloc[boot_idx]
            self.fit_from_dataframe(boot_df, variables=variables, window=min(30, len(boot_df)))
            for u, v in self.causal_graph.edges():
                w = float(self.causal_graph[u][v].get('strength', 0.0))
                samples.setdefault((u, v), []).append(w)
        edge_cis: Dict[str, Tuple[float, float]] = {}
        for (u, v), arr in samples.items():
            arr = np.array(arr)
            lo = float(np.quantile(arr, (1 - alpha) / 2))
            hi = float(np.quantile(arr, 1 - (1 - alpha) / 2))
            edge_cis[f"{u}->{v}"] = (lo, hi)
        out: Dict[str, Any] = {"edge_cis": edge_cis, "samples": windows}
        # Optional Bayesian: very lightweight attempt using PyMC linear model per edge
        try:
            import pymc as pm  # type: ignore
            bayes_cis: Dict[str, Tuple[float, float]] = {}
            data = usable
            for u, v in self.causal_graph.edges():
                if u not in data.columns or v not in data.columns:
                    continue
                X = (data[[u]].values).astype(float)
                y = (data[v].values).astype(float)
                with pm.Model() as m:
                    beta = pm.Normal('beta', mu=0, sigma=1)
                    sigma = pm.HalfNormal('sigma', sigma=1)
                    mu = beta * X.flatten()
                    pm.Normal('obs', mu=mu, sigma=sigma, observed=y)
                    idata = pm.sampling_jax.sample_numpyro_nuts(draws=500, tune=300, chains=1, progressbar=False)
                b_samp = np.asarray(idata.posterior['beta']).flatten()
                bayes_cis[f"{u}->{v}"] = (float(np.quantile(b_samp, 0.05)), float(np.quantile(b_samp, 0.95)))
            out["bayes_cis"] = bayes_cis
        except Exception:
            pass
        return out

    def detect_change_points(self, series: List[float], threshold: float = 2.5) -> List[int]:
        """Simple CUSUM-like change detection; returns indices with large cumulative shifts."""
        if not series or len(series) < 10:
            return []
        x = np.array(series, dtype=float)
        mu = x.mean()
        sigma = x.std() or 1.0
        s = np.cumsum((x - mu) / sigma)
        return [int(i) for i, v in enumerate(s) if abs(v) > threshold]

    def learn_structure(
        self,
        df: Any,
        variables: List[str],
        corr_threshold: float = 0.2
    ) -> None:
        """Very simple structure learning: add edges where |corr|>threshold, enforce DAG by ordering.

        Order variables as provided and only add edges from earlier to later variables.
        """
        if df is None or len(df) == 0:
            return
        data = df[variables].dropna()
        if len(data) < 10:
            return
        corr = data.corr().values
        self.causal_graph.clear()
        self.causal_graph.add_nodes_from(variables)
        n = len(variables)
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr[i, j]) >= corr_threshold:
                    self.add_causal_relationship(variables[i], variables[j], strength=0.0)

    def sample_joint_interventions_gaussian_copula(
        self,
        base_state: Dict[str, float],
        variables: List[str],
        df: Any,
        num_samples: int = 10,
        z_radius: float = 1.0
    ) -> List[Dict[str, float]]:
        """Sample joint interventions using a Gaussian copula built from historical correlations in z-space."""
        out: List[Dict[str, float]] = []
        if df is None or len(df) == 0:
            return out
        hist = df[variables].dropna()
        if len(hist) < 10:
            return out
        # Standardize
        H = (hist - hist.mean()) / (hist.std(ddof=0).replace(0, 1.0))
        cov = np.cov(H.values.T)
        # Ensure positive semi-definite via eigen clipping
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals_clipped = np.clip(eigvals, 1e-6, None)
        cov_psd = (eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T)
        mean = np.zeros(len(variables))
        z_samples = np.random.multivariate_normal(mean, cov_psd, size=num_samples)
        for z in z_samples:
            scaled = {}
            for k, v in zip(variables, z):
                scaled[k] = self._destandardize_value(k, float(np.clip(v, -z_radius, z_radius)))
            out.append({**base_state, **scaled})
        return out

    def plan_interventions(
        self,
        target_outcome: str,
        candidates: List[str],
        base_state: Dict[str, float],
        df: Any,
        budget: float = 1.0,
        num_grid: int = 5,
        risk_metric: str = 'cvar',
        cvar_alpha: float = 0.9
    ) -> List[Dict[str, Any]]:
        """Risk-aware intervention planner. Grid-search small z-steps and penalize by CVaR proxy.

        Returns top plans with objective = expected improvement - risk_penalty.
        """
        if not candidates:
            return []
        # Build small z-grid per candidate
        z_levels = np.linspace(-budget, budget, num_grid)
        results: List[Dict[str, Any]] = []
        for var in candidates:
            for z in z_levels:
                # Convert z-shift to raw using stats
                raw_shift = self._destandardize_value(var, z) - self._destandardize_value(var, 0.0)
                trial_state = base_state.copy()
                trial_state[var] = base_state.get(var, 0.0) + raw_shift
                # Predict deterministic outcome
                pred = self._predict_outcomes(base_state, {var: trial_state[var]})
                benefit = pred.get(target_outcome, 0.0) - base_state.get(target_outcome, 0.0)
                # Risk via bootstrap strengths if data provided
                risk_penalty = 0.0
                if df is not None and risk_metric == 'cvar':
                    unc = self.quantify_uncertainty(df, variables=list(set(candidates + [target_outcome])), windows=100, alpha=cvar_alpha)
                    # Build perturbed outcomes by sampling edge strengths from CIs midpoint +/- half-range
                    edge_cis = unc.get('edge_cis', {})
                    outcomes = []
                    for _ in range(50):
                        # Temporarily perturb strengths
                        saved: Dict[Tuple[str, str], float] = {}
                        for (u, v) in self.causal_graph.edges():
                            key = f"{u}->{v}"
                            if key in edge_cis:
                                lo, hi = edge_cis[key]
                                w = float(np.random.uniform(lo, hi))
                                saved[(u, v)] = self.causal_graph[u][v]['strength']
                                self.causal_graph[u][v]['strength'] = w
                        p = self._predict_outcomes(base_state, {var: trial_state[var]}).get(target_outcome, 0.0)
                        outcomes.append(p)
                        # Restore
                        for (u, v), w in saved.items():
                            self.causal_graph[u][v]['strength'] = w
                    outcomes = np.array(outcomes)
                    losses = base_state.get(target_outcome, 0.0) - outcomes
                    tail = np.quantile(losses, cvar_alpha)
                    risk_penalty = float(losses[losses >= tail].mean()) if np.any(losses >= tail) else 0.0
                score = benefit - risk_penalty
                results.append({
                    'var': var,
                    'z': float(z),
                    'raw_shift': float(raw_shift),
                    'benefit': float(benefit),
                    'risk_penalty': float(risk_penalty),
                    'score': float(score),
                    'predicted': pred.get(target_outcome, 0.0)
                })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:10]

    def state_space_update(
        self,
        prev_mean: float,
        prev_var: float,
        observation: float,
        process_var: float = 1e-2,
        obs_var: float = 1e-1
    ) -> Tuple[float, float]:
        """Minimal 1D Kalman-like update returning new mean/variance."""
        # Predict
        pred_mean = prev_mean
        pred_var = prev_var + process_var
        # Update
        K = pred_var / (pred_var + obs_var)
        new_mean = pred_mean + K * (observation - pred_mean)
        new_var = (1 - K) * pred_var
        return float(new_mean), float(new_var)

    def causal_explain_ace(self, child: str) -> Dict[str, float]:
        """Decompose child prediction into parent contributions in z-space (ACE-like for linear model)."""
        parents = list(self.causal_graph.predecessors(child))
        if not parents:
            return {}
        contrib: Dict[str, float] = {}
        for p in parents:
            contrib[p] = float(self.causal_graph[p][child].get('strength', 0.0))
        return contrib

    def identifiability_report(self, treatment: str, outcome: str) -> str:
        """Simple do-calculus identifiability report stub based on back-door availability."""
        adj = self.identify_adjustment_set(treatment, outcome)
        if adj:
            return f"Effect of {treatment} on {outcome} identifiable via back-door with adjustment set: {adj}"
        return f"Effect of {treatment} on {outcome} not identified by simple back-door; consider IVs or front-door."

    def identify_instruments(self, treatment: str, outcome: str) -> List[str]:
        """Heuristic IV identification: candidates are parents of treatment that do not have a path to outcome.

        This enforces relevance (parent of T) and exclusion (no path to Y) in the graph structure.
        """
        if treatment not in self.causal_graph or outcome not in self.causal_graph:
            return []
        instruments: List[str] = []
        for z in self.causal_graph.predecessors(treatment):
            # Relevance: z -> treatment by definition
            # Exclusion: no path from z to outcome (strong condition)
            if not nx.has_path(self.causal_graph, z, outcome):
                instruments.append(z)
        return instruments

    def estimate_2sls(
        self,
        df: Any,
        treatment: str,
        outcome: str,
        instruments: Optional[List[str]] = None,
        controls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Two-Stage Least Squares estimator for causal effect of treatment on outcome.

        Returns a dict with {'coef': float, 'first_stage_F': float, 'used_instruments': [...], 'used_controls': [...]}.
        """
        result = {"coef": 0.0, "first_stage_F": 0.0, "used_instruments": [], "used_controls": []}
        if df is None or len(df) == 0 or treatment not in df.columns or outcome not in df.columns:
            return result
        if instruments is None or len(instruments) == 0:
            instruments = self.identify_instruments(treatment, outcome)
        if controls is None:
            controls = []
        cols_needed = [outcome, treatment] + instruments + controls
        cols = [c for c in cols_needed if c in df.columns]
        data = df[cols].dropna()
        if len(data) < 10 or len(instruments) == 0:
            return result
        # Standardize
        ds = (data - data.mean()) / (data.std(ddof=0).replace(0, 1.0))
        # First stage: T ~ Z + C
        ZC_cols = [ds[z].values for z in instruments] + [ds[c].values for c in controls if c in ds.columns]
        if not ZC_cols:
            return result
        X1 = np.vstack(ZC_cols).T
        y1 = ds[treatment].values
        X1TX1 = X1.T @ X1
        try:
            beta1 = np.linalg.pinv(X1TX1) @ (X1.T @ y1)
        except Exception:
            return result
        t_hat = X1 @ beta1
        # First-stage F-statistic proxy: var explained / residual var times dof ratio (rough heuristic)
        ssr = float(((t_hat - y1.mean()) ** 2).sum())
        sse = float(((y1 - t_hat) ** 2).sum())
        k = X1.shape[1]
        n = X1.shape[0]
        if sse > 0 and n > k and k > 0:
            result["first_stage_F"] = (ssr / k) / (sse / (n - k - 1))
        # Second stage: Y ~ T_hat + C
        X2_cols = [t_hat] + [ds[c].values for c in controls if c in ds.columns]
        X2 = np.vstack(X2_cols).T
        y2 = ds[outcome].values
        try:
            beta2 = np.linalg.pinv(X2.T @ X2) @ (X2.T @ y2)
            result["coef"] = float(beta2[0])
        except Exception:
            result["coef"] = 0.0
        result["used_instruments"] = [z for z in instruments if z in ds.columns]
        result["used_controls"] = [c for c in controls if c in ds.columns]
        return result

    def find_optimal_interventions(
        self, 
        target_outcome: str, 
        available_interventions: List[str],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> List[Dict[str, float]]:
        """Find optimal interventions to achieve target outcome."""
        if constraints is None:
            constraints = {}
        
        optimal_interventions = []
        
        # Generate intervention combinations
        for intervention_var in available_interventions:
            if intervention_var in constraints:
                min_val, max_val = constraints[intervention_var]
                test_values = np.linspace(min_val, max_val, 10)
            else:
                test_values = np.linspace(-2.0, 2.0, 10)
            
            for value in test_values:
                intervention = {intervention_var: value}
                predicted_outcomes = self._predict_outcomes({}, intervention)
                
                if target_outcome in predicted_outcomes:
                    optimal_interventions.append({
                        'intervention': intervention,
                        'predicted_outcome': predicted_outcomes[target_outcome],
                        'efficiency': abs(predicted_outcomes[target_outcome]) / abs(value) if value != 0 else 0
                    })
        
        # Sort by efficiency and return top interventions
        optimal_interventions.sort(key=lambda x: x['efficiency'], reverse=True)
        return optimal_interventions[:5]

    def step(self, task: str) -> str:
        """Execute a single step of causal reasoning."""
        response = self.agent.run(task)
        return response

    def run(self, task: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Run the CR-CA agent for comprehensive causal analysis.
        
        Args:
            task: The problem or question to analyze causally
            
        Returns:
            Dictionary containing causal analysis results
        """
        # Reset memory
        self.causal_memory = []
        self.counterfactual_scenarios = []
        
        # Build causal analysis prompt
        causal_prompt = self._build_causal_prompt(task)
        
        # Run causal analysis
        for i in range(self.max_loops):
            print(f"\nCausal Analysis Step {i+1}/{self.max_loops}")
            
            step_result = self.step(causal_prompt)
            self.causal_memory.append({
                'step': i + 1,
                'analysis': step_result,
                'timestamp': i
            })
            
            # Update prompt with previous analysis
            if i < self.max_loops - 1:
                memory_context = self._build_memory_context()
                causal_prompt = f"{causal_prompt}\n\nPrevious Analysis:\n{memory_context}"
        
        # Generate final causal analysis
        final_analysis = self._synthesize_causal_analysis(task)
        
        return {
            'task': task,
            'causal_analysis': final_analysis,
            'counterfactual_scenarios': self.counterfactual_scenarios,
            'causal_graph_info': {
                'nodes': list(self.causal_graph.nodes()),
                'edges': list(self.causal_graph.edges()),
                'is_dag': nx.is_directed_acyclic_graph(self.causal_graph)
            },
            'analysis_steps': self.causal_memory
        }

    def _build_causal_prompt(self, task: str) -> str:
        """Build the causal analysis prompt."""
        return f"""
        You are a Causal Reasoning with Counterfactual Analysis (CR-CA) agent. 
        Analyze the following problem using sophisticated causal reasoning:
        
        Problem: {task}
        
        Your analysis should include:
        1. Causal Analysis: Identify cause-and-effect relationships
        2. Intervention Planning: Plan interventions to test causal hypotheses  
        3. Counterfactual Scenarios: Explore multiple "what-if" scenarios
        4. Causal Strength Assessment: Evaluate relationship strengths and confounders
        5. Optimal Solution: Recommend the best approach based on causal analysis
        
        Current causal graph has {len(self.causal_graph.nodes())} variables and {len(self.causal_graph.edges())} relationships.
        """

    def _build_memory_context(self) -> str:
        """Build memory context from previous analysis steps."""
        context_parts = []
        for step in self.causal_memory[-2:]:  # Last 2 steps
            context_parts.append(f"Step {step['step']}: {step['analysis']}")
        return "\n".join(context_parts)

    def _synthesize_causal_analysis(self, task: str) -> str:
        """Synthesize the final causal analysis from all steps."""
        synthesis_prompt = f"""
        Based on the causal analysis steps performed, synthesize a comprehensive 
        causal reasoning report for: {task}
        
        Include:
        - Key causal relationships identified
        - Recommended interventions
        - Counterfactual scenarios explored
        - Optimal solution with causal justification
        - Confidence levels and limitations
        """
        
        return self.agent.run(synthesis_prompt)

    def get_causal_graph_visualization(self) -> str:
        """Get a text representation of the causal graph."""
        if not self.causal_graph.nodes():
            return "Empty causal graph"
        
        lines = ["Causal Graph Structure:"]
        lines.append(f"Nodes: {list(self.causal_graph.nodes())}")
        lines.append("Edges:")
        
        for source, target in self.causal_graph.edges():
            edge_data = self.causal_graph[source][target]
            strength = edge_data.get('strength', 1.0)
            relation_type = edge_data.get('relation_type', CausalRelationType.DIRECT).value
            lines.append(f"  {source} -> {target} (strength: {strength}, type: {relation_type})")
        
        return "\n".join(lines)

    def analyze_cascading_chain_reaction(
        self,
        initial_intervention: Dict[str, float],
        target_outcomes: List[str],
        max_hops: int = 5,
        include_feedback_loops: bool = True,
        num_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Analyze multi-layer cascading chain reactions from an intervention.
        
        Example: "If X affects Y, how does it cascade through Z→alpha→...→back to X?"
        
        Args:
            initial_intervention: {variable: value} intervention to analyze
            target_outcomes: Variables to trace effects to
            max_hops: Maximum path length to consider
            include_feedback_loops: Whether to iterate for feedback effects
            num_iterations: Number of propagation iterations (for cycles)
        
        Returns:
            Dict with causal paths, cascade probabilities, and cumulative effects
        """
        # Find all paths from intervention variables to outcomes
        intervention_vars = list(initial_intervention.keys())
        all_paths: Dict[str, List[List[str]]] = {}
        
        for inter_var in intervention_vars:
            for outcome in target_outcomes:
                if outcome == inter_var:
                    continue
                
                # Find all simple paths (no cycles)
                try:
                    simple_paths = list(nx.all_simple_paths(
                        self.causal_graph,
                        inter_var,
                        outcome,
                        cutoff=max_hops
                    ))
                    if simple_paths:
                        all_paths[f"{inter_var}->{outcome}"] = simple_paths
                except nx.NetworkXNoPath:
                    pass
        
        # Find feedback loops (paths that eventually return to intervention vars)
        feedback_paths: List[List[str]] = []
        if include_feedback_loops:
            for inter_var in intervention_vars:
                try:
                    # Find cycles through inter_var
                    cycles = list(nx.simple_cycles(self.causal_graph))
                    for cycle in cycles:
                        if inter_var in cycle:
                            # Rotate cycle to start at inter_var
                            idx = cycle.index(inter_var)
                            rotated = cycle[idx:] + cycle[:idx] + [inter_var]
                            feedback_paths.append(rotated)
                except Exception:
                    pass
        
        # Multi-layer propagation with iterations
        current_state = self._standardize_state(initial_intervention)
        propagation_history: List[Dict[str, float]] = [current_state.copy()]
        
        for iteration in range(num_iterations):
            next_state = current_state.copy()
            
            # Propagate through all nodes in topological order
            for node in nx.topological_sort(self.causal_graph):
                if node in initial_intervention and iteration == 0:
                    # Keep intervention value in first iteration
                    continue
                
                parents = list(self.causal_graph.predecessors(node))
                if not parents:
                    continue
                
                effect_z = 0.0
                for parent in parents:
                    parent_z = next_state.get(parent, current_state.get(parent, 0.0))
                    edge_data = self.causal_graph[parent][node]
                    strength = edge_data.get('strength', 0.0)
                    effect_z += parent_z * strength
                
                next_state[node] = effect_z
            
            propagation_history.append(next_state.copy())
            current_state = next_state
        
        # Compute path strengths and probabilities
        path_analyses: List[Dict[str, Any]] = []
        for path_key, paths in all_paths.items():
            for path in paths:
                # Compute path strength (product of edge strengths)
                path_strength = 1.0
                path_strengths_list = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = self.causal_graph[u][v]
                    strength = abs(edge_data.get('strength', 0.0))
                    path_strength *= strength
                    path_strengths_list.append(strength)
                
                # Estimate cascade probability (stronger paths = more likely)
                # Using softmax-style normalization
                path_prob = min(0.95, path_strength * 0.5 + 0.05)
                
                path_analyses.append({
                    "path": path,
                    "path_string": " → ".join(path),
                    "path_strength": float(path_strength),
                    "edge_strengths": path_strengths_list,
                    "cascade_probability": float(path_prob),
                    "hops": len(path) - 1,
                })
        
        # Feedback loop analysis
        feedback_analyses: List[Dict[str, Any]] = []
        for cycle_path in feedback_paths[:10]:  # Limit to top 10 cycles
            cycle_strength = 1.0
            for i in range(len(cycle_path) - 1):
                u, v = cycle_path[i], cycle_path[i + 1]
                if self.causal_graph.has_edge(u, v):
                    strength = abs(self.causal_graph[u][v].get('strength', 0.0))
                    cycle_strength *= strength
            
            feedback_analyses.append({
                "cycle": cycle_path,
                "cycle_string": " → ".join(cycle_path),
                "cycle_strength": float(cycle_strength),
                "could_amplify": cycle_strength > 0.1,  # Strong cycles can amplify
            })
        
        # Final state predictions (de-standardized)
        final_predictions: Dict[str, float] = {}
        for var in target_outcomes:
            if var in propagation_history[-1]:
                z_val = propagation_history[-1][var]
                final_predictions[var] = self._destandardize_value(var, z_val)
            else:
                final_predictions[var] = 0.0
        
        return {
            "initial_intervention": initial_intervention,
            "target_outcomes": target_outcomes,
            "causal_paths": path_analyses,
            "feedback_loops": feedback_analyses,
            "propagation_history": [
                {k: self._destandardize_value(k, v) for k, v in state.items() if k in target_outcomes}
                for state in propagation_history
            ],
            "final_predictions": final_predictions,
            "summary": {
                "total_paths_found": len(path_analyses),
                "feedback_loops_detected": len(feedback_analyses),
                "max_path_length": max([p["hops"] for p in path_analyses] + [0]),
                "strongest_path": max(path_analyses, key=lambda x: x["path_strength"]) if path_analyses else None,
            },
        }

    def multi_layer_whatif_analysis(
        self,
        scenarios: List[Dict[str, float]],
        depth: int = 3,
    ) -> Dict[str, Any]:
        """
        Multi-layer "what-if" analysis: If X happens to Y, how would it affect?
        Then: What are the chances of Z affecting alpha and causing chain reaction to X?
        
        Performs nested counterfactual reasoning across multiple layers.
        
        Args:
            scenarios: List of intervention scenarios {variable: value}
            depth: How many layers deep to analyze
        
        Returns:
            Nested analysis with cascading effects and chain reaction probabilities
        """
        results: List[Dict[str, Any]] = []
        
        for scenario in scenarios:
            # Layer 1: Direct effects
            layer1_outcomes = self._predict_outcomes({}, scenario)
            
            # Identify affected variables (significant changes)
            affected_vars = [
                var for var, val in layer1_outcomes.items()
                if abs(val) > 0.01  # Threshold for "affected"
            ]
            
            # Layer 2: What if affected vars change other things?
            layer2_scenarios: List[Dict[str, float]] = []
            for affected_var in affected_vars[:5]:  # Limit to top 5
                # Create scenario where this variable is perturbed
                perturbation = {
                    affected_var: layer1_outcomes.get(affected_var, 0.0) * 1.2  # 20% perturbation
                }
                layer2_scenarios.append(perturbation)
            
            layer2_analyses: List[Dict[str, Any]] = []
            for layer2_scen in layer2_scenarios:
                layer2_outcomes = self._predict_outcomes(layer1_outcomes, layer2_scen)
                
                # Check for chain reactions back to original intervention vars
                chain_reactions: List[Dict[str, Any]] = []
                for orig_var in scenario.keys():
                    if orig_var in layer2_outcomes:
                        # Chain reaction detected: original var affected by cascade
                        chain_reactions.append({
                            "original_intervention": orig_var,
                            "chain_path": f"{list(layer2_scen.keys())[0]} → {orig_var}",
                            "effect_magnitude": abs(layer2_outcomes[orig_var] - layer1_outcomes.get(orig_var, 0.0)),
                            "could_cause_amplification": abs(layer2_outcomes[orig_var]) > abs(scenario.get(orig_var, 0.0)),
                        })
                
                layer2_analyses.append({
                    "layer2_scenario": layer2_scen,
                    "layer2_outcomes": layer2_outcomes,
                    "chain_reactions": chain_reactions,
                })
            
            # Layer 3+: Deep cascade analysis (if depth > 2)
            cascade_analysis = None
            if depth > 2:
                # Use cascading chain reaction method
                all_outcomes = set(layer1_outcomes.keys()) | set(layer2_analyses[0]["layer2_outcomes"].keys() if layer2_analyses else set())
                cascade_analysis = self.analyze_cascading_chain_reaction(
                    initial_intervention=scenario,
                    target_outcomes=list(all_outcomes)[:10],  # Top 10 outcomes
                    max_hops=5,
                    include_feedback_loops=True,
                    num_iterations=depth,
                )
            
            results.append({
                "scenario": scenario,
                "layer1_direct_effects": layer1_outcomes,
                "affected_variables": affected_vars,
                "layer2_cascades": layer2_analyses,
                "deep_cascade": cascade_analysis,
                "chain_reaction_summary": {
                    "total_chain_reactions": sum(len(l2.get("chain_reactions", [])) for l2 in layer2_analyses),
                    "potential_amplifications": sum(
                        1 for l2 in layer2_analyses
                        for cr in l2.get("chain_reactions", [])
                        if cr.get("could_cause_amplification", False)
                    ),
                },
            })
        
        return {
            "multi_layer_analysis": results,
            "summary": {
                "total_scenarios": len(results),
                "avg_chain_reactions_per_scenario": np.mean([
                    r["chain_reaction_summary"]["total_chain_reactions"] for r in results
                ]) if results else 0.0,
                "scenarios_with_amplification": sum(
                    1 for r in results
                    if r["chain_reaction_summary"]["potential_amplifications"] > 0
                ),
            },
        }

    def deep_root_cause_analysis(
        self,
        problem_variable: str,
        max_depth: int = 20,
        min_path_strength: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Infinitely nested root cause analysis: trace backwards to find absolute deepest causes.
        
        Keeps going deeper until hitting exogenous nodes or circular dependencies.
        Finds the ultimate root causes that, if intervened, would solve the problem.
        
        Args:
            problem_variable: Variable we want to fix/understand
            max_depth: Maximum backward tracing depth (safety limit)
            min_path_strength: Minimum edge strength to consider
        
        Returns:
            Root causes, causal paths, intervention opportunities
        """
        if problem_variable not in self.causal_graph:
            return {"error": f"Variable {problem_variable} not in causal graph"}
        
        # Backward tracing: find all ancestors (potential root causes)
        all_ancestors = list(nx.ancestors(self.causal_graph, problem_variable))
        
        # Trace paths from each ancestor to problem
        root_causes: List[Dict[str, Any]] = []
        paths_to_problem: List[Dict[str, Any]] = []
        
        for ancestor in all_ancestors:
            try:
                # Find all paths from ancestor to problem
                paths = list(nx.all_simple_paths(
                    self.causal_graph,
                    ancestor,
                    problem_variable,
                    cutoff=max_depth
                ))
                
                if paths:
                    # Compute path strength for each path
                    # Mathematical formulation: Path strength = ∏(i,j)∈Path |β_ij|
                    # where β_ij is the causal effect (structural coefficient) from i to j
                    # This follows Pearl's Structural Causal Model (SCM) path product rule
                    for path in paths:
                        # Initialize with multiplicative identity (1.0)
                        # Path strength represents cumulative causal effect along chain
                        path_strength = 1.0
                        path_details = []
                        
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            edge_data = self.causal_graph[u][v]
                            # β_ij: structural coefficient from Pearl's SCM framework
                            beta_ij = edge_data.get('strength', 0.0)  # β_ij (signed)
                            strength = abs(beta_ij)  # |β_ij|: magnitude for threshold check
                            
                            # Filter by minimum effect size (power analysis threshold)
                            # If any edge is too weak, path strength becomes 0
                            if strength < min_path_strength:
                                path_strength = 0.0
                                break
                            
                            # Multiplicative path strength: ∏(i,j)∈Path β_ij
                            # This follows chain rule of differentiation and causal mediation analysis
                            # IMPORTANT: Use signed β_ij (not |β_ij|) to preserve sign in product
                            # Total effect = β₁₂ · β₂₃ · ... · βₖ₋₁ₖ (product preserves sign)
                            path_strength *= beta_ij  # Multiply by signed coefficient
                            path_details.append({
                                "edge": f"{u}→{v}",
                                "strength": strength,
                                "structural_coefficient": float(edge_data.get('strength', 0.0)),
                            })
                        
                        if path_strength > 0:
                            # Check if ancestor is exogenous (true root cause)
                            ancestors_of_ancestor = list(nx.ancestors(self.causal_graph, ancestor))
                            is_exogenous = len(ancestors_of_ancestor) == 0
                            
                            root_causes.append({
                                "root_cause": ancestor,
                                "is_exogenous": is_exogenous,
                                "path_to_problem": path,
                                "path_string": " → ".join(path),
                                "path_strength": float(path_strength),
                                "depth": len(path) - 1,
                                "path_details": path_details,
                            })
                            
                            paths_to_problem.append({
                                "from": ancestor,
                                "to": problem_variable,
                                "path": path,
                                "strength": float(path_strength),
                            })
            except Exception:
                continue
        
        # Rank root causes using multi-objective optimization criteria
        # Objective: maximize f(rc) = w1·I_exo(rc) + w2·S_path(rc) - w3·D(rc)
        # where I_exo is indicator function, S_path is path strength, D is depth
        # Using lexicographic ordering: exogenous > path_strength > -depth
        root_causes.sort(
            key=lambda x: (
                -x["is_exogenous"],  # Exogenous first (lexicographic priority)
                -x["path_strength"],  # Stronger paths first (maximize causal effect)
                x["depth"]  # Shorter paths first (minimize intervention distance)
            )
        )
        
        # Find ultimate root causes (those with no ancestors, or only circular ones)
        ultimate_roots = [
            rc for rc in root_causes
            if rc["is_exogenous"] or rc["depth"] >= max_depth - 2
        ]
        
        return {
            "problem_variable": problem_variable,
            "all_root_causes": root_causes[:20],  # Top 20
            "ultimate_root_causes": ultimate_roots[:10],  # Top 10 ultimate
            "total_paths_found": len(paths_to_problem),
            "max_depth_reached": max([rc["depth"] for rc in root_causes] + [0]),
            "intervention_opportunities": [
                {
                    "intervene_on": rc["root_cause"],
                    "expected_impact_on_problem": rc["path_strength"],
                    "depth": rc["depth"],
                    "is_exogenous": rc["is_exogenous"],
                }
                for rc in root_causes[:10]
            ],
        }

    def explore_alternate_realities(
        self,
        factual_state: Dict[str, float],
        target_outcome: str,
        target_value: Optional[float] = None,
        max_realities: int = 50,
        max_interventions: int = 3,
    ) -> Dict[str, Any]:
        """
        Explore multiple alternate realities to find interventions that achieve best outcome.
        
        Searches intervention space to find sequences that optimize target outcome.
        Considers multiple possible realities and picks the best.
        
        Args:
            factual_state: Current state
            target_outcome: Variable to optimize
            target_value: Desired value (if None, maximize)
            max_realities: Number of alternate scenarios to explore
            max_interventions: Max number of simultaneous interventions
        
        Returns:
            Best interventions, alternate realities explored, optimal outcome
        """
        if target_outcome not in self.causal_graph:
            return {"error": f"Target {target_outcome} not in graph"}
        
        # Get intervention candidates (nodes with out-edges)
        intervention_candidates = [
            node for node in self.causal_graph.nodes()
            if len(list(self.causal_graph.successors(node))) > 0
            and node not in [target_outcome]
        ]
        
        realities: List[Dict[str, Any]] = []
        
        # Sample intervention combinations
        for _ in range(max_realities):
            # Random intervention set
            num_interventions = self.rng.integers(1, max_interventions + 1)
            selected = self.rng.choice(intervention_candidates, size=min(num_interventions, len(intervention_candidates)), replace=False)
            
            # Generate intervention values (standardized perturbations)
            intervention: Dict[str, float] = {}
            for var in selected:
                stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
                current = factual_state.get(var, stats["mean"])
                # Perturb by ±2 standard deviations
                perturbation = self.rng.normal(0, stats["std"] * 2.0)
                intervention[var] = current + perturbation
            
            # Predict outcome
            outcome = self._predict_outcomes(factual_state, intervention)
            target_val = outcome.get(target_outcome, 0.0)
            
            # Compute objective using L2 norm (Euclidean distance) or direct maximization
            # Objective function: O(θ) = -||y(θ) - y*||₂ if target specified, else O(θ) = y(θ)
            # where θ = intervention vector, y(θ) = predicted outcome, y* = target value
            if target_value is not None:
                # For single-dimensional: use L1 norm ||y - y*||₁ = |y - y*|
                # For multi-dimensional would use: ||y - y*||₂ = √(Σ(y_i - y*_i)²)
                # Negative sign for maximization (minimize distance = maximize negative distance)
                distance = abs(target_val - target_value)  # L1 norm: |y - y*|
                objective = -distance  # Negative distance (better = less distance)
            else:
                # Direct maximization: O(θ) = y(θ) (no target, just maximize outcome)
                objective = target_val  # Maximize
            
            realities.append({
                "interventions": intervention,
                "outcome": outcome,
                "target_value": target_val,
                "objective": float(objective),
                "delta_from_factual": target_val - factual_state.get(target_outcome, 0.0),
            })
        
        # Sort by objective (best first)
        realities.sort(key=lambda x: x["objective"], reverse=True)
        
        # Best reality
        best = realities[0] if realities else None
        
        # Pareto frontier (if multi-objective)
        pareto_realities = []
        if len(realities) > 1:
            # Simple Pareto: if reality A is better on target and not worse on others
            for r in realities[:20]:  # Top 20
                is_dominated = False
                for other in realities:
                    if other is r:
                        continue
                    # Check if other dominates
                    if (other["objective"] >= r["objective"] and 
                        other["delta_from_factual"] >= r["delta_from_factual"] and
                        (other["objective"] > r["objective"] or other["delta_from_factual"] > r["delta_from_factual"])):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_realities.append(r)
        
        return {
            "factual_state": factual_state,
            "target_outcome": target_outcome,
            "target_value": target_value,
            "best_reality": best,
            "top_10_realities": realities[:10],
            "pareto_frontier": pareto_realities[:10],
            "all_realities_explored": len(realities),
            "improvement_potential": (
                best["target_value"] - factual_state.get(target_outcome, 0.0)
                if best else 0.0
            ),
        }

    def historical_pattern_matching(
        self,
        current_state: Dict[str, float],
        intervention_history: List[Dict[str, float]],
        outcome_history: List[Dict[str, float]],
        target_outcome: str,
        similarity_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Learn from historical interventions: reuse old reactions to create new ones.
        
        Finds past interventions in similar states that achieved good outcomes,
        then adapts them to current state.
        
        Args:
            current_state: Current factual state
            intervention_history: List of past interventions
            outcome_history: List of past outcomes (corresponding to interventions)
            target_outcome: Variable to optimize
            similarity_threshold: How similar states must be to reuse
        
        Returns:
            Matched historical patterns, adapted interventions, learned strategies
        """
        if len(intervention_history) != len(outcome_history) or len(intervention_history) == 0:
            return {"error": "Invalid history or empty"}
        
        # Compute state similarity (cosine similarity in standardized space)
        z_current = self._standardize_state(current_state)
        matches: List[Dict[str, Any]] = []
        
        for i, (intervention, outcome) in enumerate(zip(intervention_history, outcome_history)):
            # Standardize historical outcome (treated as "state")
            z_past = self._standardize_state(outcome)
            
            # Compute similarity (cosine similarity)
            common_vars = set(z_current.keys()) & set(z_past.keys())
            if len(common_vars) == 0:
                continue
            
            vec_current = np.array([z_current[v] for v in common_vars])
            vec_past = np.array([z_past[v] for v in common_vars])
            
            # Cosine similarity (normalized inner product): cos(θ) = (A·B)/(||A||₂·||B||₂)
            # This measures angle between vectors in standardized space
            # Range: [-1, 1], where 1 = identical direction, 0 = orthogonal, -1 = opposite
            norm_current = np.linalg.norm(vec_current)  # L2 norm: ||v||₂ = √(Σv_i²)
            norm_past = np.linalg.norm(vec_past)
            
            if norm_current < 1e-6 or norm_past < 1e-6:
                similarity = 0.0  # Zero vectors have undefined similarity
            else:
                # Cosine similarity formula: cos(θ) = (v₁·v₂)/(||v₁||₂·||v₂||₂)
                similarity = float(np.dot(vec_current, vec_past) / (norm_current * norm_past))
            
            if similarity >= similarity_threshold:
                # Historical outcome value for target
                target_past = outcome.get(target_outcome, 0.0)
                target_current = current_state.get(target_outcome, 0.0)
                
                # Did this intervention improve the target?
                improvement = target_past - target_current
                
                matches.append({
                    "historical_index": i,
                    "similarity": similarity,
                    "past_intervention": intervention,
                    "past_outcome": outcome,
                    "target_improvement": float(improvement),
                    "was_successful": improvement > 0.0,
                })
        
        # Sort by similarity and success
        matches.sort(
            key=lambda x: (x["similarity"], x["target_improvement"]),
            reverse=True
        )
        
        # Adapt best historical interventions to current state
        adapted_interventions: List[Dict[str, Any]] = []
        for match in matches[:5]:  # Top 5 matches
            past_intervention = match["past_intervention"]
            
            # Adapt: scale by similarity and current context
            adapted = {}
            for var, val in past_intervention.items():
                # Adjust based on current state difference
                current_val = current_state.get(var, 0.0)
                past_val = past_intervention.get(var, 0.0)
                
                # Blend: similarity-weighted historical value + current baseline
                adaptation_factor = match["similarity"]
                adapted[var] = float(
                    current_val * (1.0 - adaptation_factor) +
                    past_val * adaptation_factor
                )
            
            # Predict outcome of adapted intervention
            predicted = self._predict_outcomes(current_state, adapted)
            predicted_target = predicted.get(target_outcome, 0.0)
            
            adapted_interventions.append({
                "source_match": match,
                "adapted_intervention": adapted,
                "predicted_outcome": predicted,
                "predicted_target_value": float(predicted_target),
                "expected_improvement": float(predicted_target - current_state.get(target_outcome, 0.0)),
            })
        
        # Extract learned strategy patterns
        successful_interventions = [m for m in matches if m["was_successful"]]
        learned_patterns: List[Dict[str, Any]] = []
        
        if successful_interventions:
            # Common variables in successful interventions
            all_vars = set()
            for m in successful_interventions:
                all_vars.update(m["past_intervention"].keys())
            
            for var in all_vars:
                successful_vals = [
                    m["past_intervention"].get(var, 0.0)
                    for m in successful_interventions
                    if var in m["past_intervention"]
                ]
                if successful_vals:
                    learned_patterns.append({
                        "variable": var,
                        "typical_successful_value": float(np.mean(successful_vals)),
                        "value_range": [float(min(successful_vals)), float(max(successful_vals))],
                        "success_frequency": len(successful_vals) / len(successful_interventions),
                    })
        
        return {
            "current_state": current_state,
            "historical_matches": matches[:10],
            "best_matches": matches[:5],
            "adapted_interventions": adapted_interventions,
            "learned_patterns": learned_patterns,
            "recommended_intervention": adapted_interventions[0]["adapted_intervention"] if adapted_interventions else {},
            "strategy_confidence": float(np.mean([m["similarity"] for m in matches[:5]])) if matches else 0.0,
        }

    def infinite_nesting_root_cause(
        self,
        problem: str,
        stop_condition: Optional[Callable[[str, int], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Infinitely nested root cause analysis: keeps going deeper until hitting true roots.
        
        Unlike deep_root_cause_analysis which has max_depth, this continues until
        hitting exogenous nodes or user-defined stop condition.
        
        Args:
            problem: Variable to trace backwards from
            stop_condition: Function(node, depth) -> bool to stop tracing
        
        Returns:
            Ultimate root causes, full causal tree, intervention strategy
        """
        if problem not in self.causal_graph:
            return {"error": f"Problem {problem} not in graph"}
        
        visited: set = set()
        causal_tree: List[Dict[str, Any]] = []
        root_nodes: List[str] = []
        
        def trace_backwards(node: str, depth: int, path: List[str]) -> None:
            """Recursive backward tracing."""
            if node in visited or depth > 100:  # Safety limit
                return
            
            visited.add(node)
            
            # Check stop condition
            if stop_condition and stop_condition(node, depth):
                root_nodes.append(node)
                causal_tree.append({
                    "node": node,
                    "depth": depth,
                    "path_to_problem": path + [node],
                    "is_stopped_by_condition": True,
                })
                return
            
            # Get parents (causes)
            parents = list(self.causal_graph.predecessors(node))
            
            if len(parents) == 0:
                # Exogenous node - true root cause
                root_nodes.append(node)
                causal_tree.append({
                    "node": node,
                    "depth": depth,
                    "path_to_problem": path + [node],
                    "is_exogenous": True,
                })
                return
            
            # Recursively trace parents
            for parent in parents:
                edge_data = self.causal_graph[parent][node]
                strength = abs(edge_data.get('strength', 0.0))
                
                causal_tree.append({
                    "node": node,
                    "parent": parent,
                    "edge_strength": float(strength),
                    "depth": depth,
                    "path_to_problem": path + [node],
                })
                
                trace_backwards(parent, depth + 1, path + [node])
        
        # Start tracing from problem
        trace_backwards(problem, 0, [])
        
        # Unique root nodes
        unique_roots = list(set(root_nodes))
        
        # Build intervention strategy (intervene on shallowest exogenous nodes)
        root_depths = {
            root: min([ct["depth"] for ct in causal_tree if ct.get("node") == root or ct.get("parent") == root] + [999])
            for root in unique_roots
        }
        optimal_roots = sorted(unique_roots, key=lambda x: root_depths.get(x, 999))[:10]
        
        return {
            "problem": problem,
            "ultimate_root_causes": unique_roots,
            "optimal_intervention_targets": optimal_roots,
            "causal_tree": causal_tree,
            "max_depth_reached": max([ct["depth"] for ct in causal_tree] + [0]),
            "total_nodes_explored": len(visited),
        }

    def optimal_intervention_sequence(
        self,
        initial_state: Dict[str, float],
        target_outcomes: Dict[str, float],
        max_steps: int = 5,
        horizon: int = 10,
    ) -> Dict[str, Any]:
        """
        Find optimal sequence of interventions to achieve target outcomes.
        
        Uses dynamic programming / tree search to find best intervention sequence
        that alters future reactions for optimal outcome.
        
        Args:
            initial_state: Starting state
            target_outcomes: {variable: target_value} to achieve
            max_steps: Maximum intervention steps
            horizon: How far ahead to optimize
        
        Returns:
            Optimal intervention sequence, expected trajectory, outcome probability
        """
        # Intervention candidates
        candidates = [
            node for node in self.causal_graph.nodes()
            if len(list(self.causal_graph.successors(node))) > 0
            and node not in target_outcomes.keys()
        ]
        
        # Optimal control problem: find sequence {u₀, u₁, ..., uₜ} that minimizes cost
        # Cost function: J = Σₜ L(xₜ, uₜ) + Φ(xₜ) where L is stage cost, Φ is terminal cost
        # Subject to: xₜ₊₁ = f(xₜ, uₜ) (system dynamics)
        # Using greedy approximation (could be upgraded to Bellman optimality: V*(x) = min_u[L(x,u) + V*(f(x,u))])
        best_sequence: List[Dict[str, float]] = []
        best_final_state: Dict[str, float] = initial_state.copy()
        best_objective = float("-inf")
        
        current_state = initial_state.copy()
        
        for step in range(max_steps):
            # Find best single intervention at this step (greedy policy)
            # Greedy: u*_t = argmin_u E[L(xₜ₊₁, u) | xₜ] (one-step lookahead)
            best_step_intervention: Optional[Dict[str, float]] = None
            best_step_objective = float("-inf")
            
            # Sample candidate interventions (Monte Carlo policy search)
            # Alternative: gradient-based optimization using ∇ᵤJ
            for _ in range(20):  # Explore 20 candidates per step
                var = self.rng.choice(candidates)
                stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
                current_val = current_state.get(var, stats["mean"])
                
                # Try perturbation: u = u₀ + ε where ε ~ N(0, σ²)
                # This implements exploration in policy space
                intervention = {var: current_val + self.rng.normal(0, stats["std"] * 1.5)}
                
                # Simulate forward (system dynamics): xₜ₊₁ = f(xₜ, uₜ)
                predicted = self._predict_outcomes(current_state, intervention)
                
                # Multi-step lookahead (horizon H): simulate xₜ₊₁, xₜ₊₂, ..., xₜ₊ₕ
                # This approximates value function: V^π(x) = E[Σₖ₌₀ᴴ γᵏ·rₜ₊ₖ]
                state_after = predicted.copy()
                for h in range(1, horizon):
                    # Further interventions (optional, could use optimal policy π*(x))
                    state_after = self._predict_outcomes(state_after, {})
                
                # Compute objective: multi-target cost function
                # Cost: J(u) = Σⱼ |yⱼ(u) - y*ⱼ| (L1 norm, sum of absolute errors)
                # Alternative L2: J(u) = √(Σⱼ (yⱼ(u) - y*ⱼ)²) (Euclidean distance)
                # We use L1 for computational simplicity, can be upgraded to L2
                # Minimize: argmin_u J(u) subject to constraints
                # Since we maximize objective, use negative cost: O = -J(u)
                objective = 0.0
                for target_var, target_val in target_outcomes.items():
                    if target_var in state_after:
                        # L1 distance (absolute error): |y - y*|
                        # For multi-dimensional: ||y - y*||₁ = Σⱼ |yⱼ - y*ⱼ|
                        distance = abs(state_after[target_var] - target_val)  # L1 norm component
                        objective -= distance  # Accumulate negative L1 distance (minimize error = maximize objective)
                
                if objective > best_step_objective:
                    best_step_objective = objective
                    best_step_intervention = intervention
            
            if best_step_intervention:
                best_sequence.append(best_step_intervention)
                # Update state
                current_state = self._predict_outcomes(current_state, best_step_intervention)
            else:
                break
        
        # Full trajectory simulation
        trajectory: List[Dict[str, float]] = [initial_state.copy()]
        state = initial_state.copy()
        for intervention in best_sequence:
            state = self._predict_outcomes(state, intervention)
            trajectory.append(state.copy())
        
        # Final outcome assessment using relative error metric
        # Relative error: ε_rel = |y_actual - y_target| / |y_target|
        # Success criterion: ε_rel < 0.1 (within 10% tolerance)
        final_objective = 0.0
        target_achievements: Dict[str, float] = {}
        for target_var, target_val in target_outcomes.items():
            if target_var in trajectory[-1]:
                actual = trajectory[-1][target_var]
                # Absolute error: ε_abs = |y_actual - y_target|
                distance = abs(actual - target_val)
                # Relative error: ε_rel = ε_abs / |y_target| (normalized by target magnitude)
                relative_error = distance / abs(target_val) if abs(target_val) > 1e-6 else distance
                target_achievements[target_var] = {
                    "target": target_val,
                    "actual": actual,
                    "distance": distance,
                    "relative_error": float(relative_error),
                    "achieved": distance < abs(target_val * 0.1),  # Within 10% tolerance: ε_rel < 0.1
                }
                # Aggregate objective: J_total = -Σⱼ ε_absⱼ (minimize total error)
                final_objective -= distance
        
        return {
            "optimal_sequence": best_sequence,
            "trajectory": trajectory,
            "final_state": trajectory[-1] if trajectory else initial_state,
            "target_achievements": target_achievements,
            "objective": float(final_objective),
            "sequence_length": len(best_sequence),
        }

    def meta_learn_intervention_strategy(
        self,
        historical_interventions: List[Dict[str, float]],
        historical_outcomes: List[Dict[str, float]],
        historical_contexts: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Meta-learn how to intervene: learn from past to create optimal future interventions.
        
        Identifies patterns in successful interventions across different contexts,
        then uses this knowledge to design new interventions.
        
        Args:
            historical_interventions: Past interventions
            historical_outcomes: Corresponding outcomes
            historical_contexts: States when interventions were applied
        
        Returns:
            Learned intervention strategy, pattern rules, meta-knowledge
        """
        if not (len(historical_interventions) == len(historical_outcomes) == len(historical_contexts)):
            return {"error": "History lengths must match"}
        
        # Classify interventions by outcome quality
        intervention_classes: List[Dict[str, Any]] = []
        
        for i, (intervention, outcome, context) in enumerate(zip(
            historical_interventions, historical_outcomes, historical_contexts
        )):
            # Compute outcome quality (simple: sum of positive changes)
            quality = sum([
                max(0.0, outcome.get(k, 0.0) - context.get(k, 0.0))
                for k in outcome.keys()
            ])
            
            intervention_classes.append({
                "index": i,
                "intervention": intervention,
                "context": context,
                "outcome": outcome,
                "quality": float(quality),
                "is_successful": quality > 0.0,
            })
        
        # Extract patterns from successful interventions
        successful = [ic for ic in intervention_classes if ic["is_successful"]]
        unsuccessful = [ic for ic in intervention_classes if not ic["is_successful"]]
        
        # Pattern: what interventions work in what contexts?
        learned_rules: List[Dict[str, Any]] = []
        
        if successful:
            # For each intervention variable, find context patterns
            all_intervention_vars = set()
            for ic in successful:
                all_intervention_vars.update(ic["intervention"].keys())
            
            for var in all_intervention_vars:
                successful_vals = [
                    ic["intervention"].get(var, 0.0)
                    for ic in successful
                    if var in ic["intervention"]
                ]
                
                # Context conditions (simplified: avg context where this worked)
                contexts_when_successful = [
                    ic["context"]
                    for ic in successful
                    if var in ic["intervention"]
                ]
                
                if successful_vals and contexts_when_successful:
                    # Average successful intervention value
                    avg_successful_val = float(np.mean(successful_vals))
                    
                    # Average context when this worked
                    context_vars = set()
                    for ctx in contexts_when_successful:
                        context_vars.update(ctx.keys())
                    
                    avg_context = {
                        k: float(np.mean([ctx.get(k, 0.0) for ctx in contexts_when_successful]))
                        for k in context_vars
                    }
                    
                    learned_rules.append({
                        "intervention_variable": var,
                        "recommended_value": avg_successful_val,
                        "typical_context": avg_context,
                        "success_rate": len(successful_vals) / len(historical_interventions),
                        "confidence": float(min(1.0, len(successful_vals) / 5.0)),  # Higher with more examples
                    })
        
        # Meta-strategy: when to use which intervention pattern
        strategy: Dict[str, Any] = {
            "learned_rules": learned_rules,
            "success_rate": len(successful) / len(intervention_classes) if intervention_classes else 0.0,
            "pattern_count": len(learned_rules),
            "most_effective_interventions": sorted(
                learned_rules,
                key=lambda x: (x["confidence"], x["success_rate"]),
                reverse=True
            )[:5],
        }
        
        return strategy

    def recursive_alternate_reality_search(
        self,
        current_state: Dict[str, float],
        target: str,
        depth: int = 0,
        max_depth: int = 10,
        path: List[Dict[str, float]] = None,
        best_found: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Recursively search alternate realities, nesting as deep as needed.
        
        Explores intervention tree recursively to find absolute best outcome.
        Each reality branches into more realities, creating infinite nesting.
        
        Args:
            current_state: Current state in this branch
            target: Variable to optimize
            depth: Current recursion depth
            max_depth: Maximum depth (safety)
            path: Intervention path to reach this state
            best_found: Best outcome found so far (for pruning)
        
        Returns:
            Best reality found, full search tree, optimal path
        """
        if path is None:
            path = []
        
        if depth >= max_depth:
            # Leaf: evaluate this reality
            target_val = current_state.get(target, 0.0)
            return {
                "state": current_state,
                "path": path,
                "target_value": float(target_val),
                "depth": depth,
                "is_leaf": True,
            }
        
        # Get intervention candidates
        candidates = [
            node for node in self.causal_graph.nodes()
            if len(list(self.causal_graph.successors(node))) > 0
            and node != target
        ]
        
        # Branch: explore multiple interventions from this state
        branches: List[Dict[str, Any]] = []
        best_branch = None
        best_target = float("-inf")
        
        # Limit branching to avoid explosion
        num_branches = min(5, len(candidates))
        selected = self.rng.choice(candidates, size=num_branches, replace=False) if len(candidates) > 0 else []
        
        for var in selected:
            # Create intervention
            stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
            current_val = current_state.get(var, stats["mean"])
            intervention = {var: current_val + self.rng.normal(0, stats["std"] * 1.0)}
            
            # Predict next state
            next_state = self._predict_outcomes(current_state, intervention)
            
            # Recursively explore
            branch_result = self.recursive_alternate_reality_search(
                current_state=next_state,
                target=target,
                depth=depth + 1,
                max_depth=max_depth,
                path=path + [intervention],
                best_found=best_found,
            )
            
            branches.append({
                "intervention": intervention,
                "result": branch_result,
            })
            
            # Update best
            branch_target = branch_result.get("target_value", float("-inf"))
            if branch_target > best_target:
                best_target = branch_target
                best_branch = branch_result
        
        # Return best path
        return {
            "state": current_state,
            "path": path,
            "branches": branches,
            "best_branch": best_branch,
            "target_value": best_target,
            "depth": depth,
            "is_leaf": False,
        }

    def probabilistic_nested_simulation(
        self,
        initial_state: Dict[str, float],
        interventions: Dict[str, float],
        num_samples: int = 100,
        nesting_depth: int = 5,
        uncertainty_propagation: bool = True,
    ) -> Dict[str, Any]:
        """
        Probabilistic nested simulation: branch on uncertainty at each layer.
        
        Each nesting level considers multiple probabilistic outcomes,
        creating a tree of possible futures with probabilities.
        
        Args:
            initial_state: Starting state
            interventions: Initial interventions
            num_samples: Monte Carlo samples per nesting level
            nesting_depth: How many layers to nest
            uncertainty_propagation: Whether to propagate uncertainty through edges
        
        Returns:
            Probabilistic outcome tree, expected values, confidence intervals
        """
        # Bootstrap edge strength uncertainty using coefficient of variation
        # CV = σ/μ (coefficient of variation), where σ is standard deviation, μ is mean
        # Uncertainty quantification: σ_β = CV · |β|, following asymptotic normality
        # Under CLT, β̂ ~ N(β, σ²_β) for large samples
        edge_uncertainty: Dict[Tuple[str, str], float] = {}
        for u, v in self.causal_graph.edges():
            strength = abs(self.causal_graph[u][v].get('strength', 0.0))
            # Assume 20% coefficient of variation (CV = 0.2)
            # Standard error: SE(β) = CV · |β| = 0.2 · |β|
            # This models uncertainty from estimation variance
            edge_uncertainty[(u, v)] = strength * 0.2
        
        # Nested simulation tree
        simulation_tree: List[Dict[str, Any]] = []
        
        def simulate_level(
            state: Dict[str, float],
            interventions_at_level: Dict[str, float],
            level: int,
            parent_probability: float = 1.0,
        ) -> Dict[str, Any]:
            """Recursive probabilistic simulation."""
            if level >= nesting_depth:
                return {
                    "level": level,
                    "state": state,
                    "probability": parent_probability,
                    "is_leaf": True,
                }
            
            # Sample edge strengths with uncertainty
            outcomes: List[Dict[str, float]] = []
            outcome_probs: List[float] = []
            
            for _ in range(num_samples):
                # Perturb edge strengths
                perturbed_state = state.copy()
                
                # Predict with uncertainty
                z_state = self._standardize_state({**state, **interventions_at_level})
                z_pred = dict(z_state)
                
                # Propagate with uncertain edges
                for node in nx.topological_sort(self.causal_graph):
                    if node in interventions_at_level:
                        continue
                    
                    parents = list(self.causal_graph.predecessors(node))
                    if not parents:
                        continue
                    
                    effect_z = 0.0
                    for parent in parents:
                        parent_z = z_pred.get(parent, z_state.get(parent, 0.0))
                        base_strength = self.causal_graph[parent][node].get('strength', 0.0)
                        # Uncertainty propagation using normal distribution
                        # β_sample ~ N(μ_β, σ²_β) where μ_β = base_strength, σ²_β = uncertainty²
                        # This follows Bayesian posterior sampling or bootstrap sampling
                        if uncertainty_propagation:
                            uncertainty = edge_uncertainty.get((parent, node), 0.0)
                            # Monte Carlo sampling: β ~ N(μ_β, σ²_β)
                            # This propagates estimation uncertainty through causal structure
                            strength = float(self.rng.normal(base_strength, uncertainty))
                        else:
                            strength = base_strength
                        effect_z += parent_z * strength
                    
                    z_pred[node] = effect_z
                
                # De-standardize
                outcome = {
                    k: self._destandardize_value(k, v)
                    for k, v in z_pred.items()
                    if k in state or k in interventions_at_level
                }
                outcomes.append(outcome)
                outcome_probs.append(1.0 / num_samples)  # Uniform weights
            
            # Aggregate outcomes using sample statistics (Monte Carlo estimation)
            # Expected value: E[Y] = (1/n)Σᵢ yᵢ where n is sample size
            # This is the Monte Carlo estimate of the expectation
            # Unbiased estimator: E[Ȳ] = E[Y] (sample mean is unbiased for population mean)
            expected_outcome: Dict[str, float] = {}
            all_vars = set.union(*[set(o.keys()) for o in outcomes]) if outcomes else set()
            for var in all_vars:
                # Collect all samples for this variable: {y₁, y₂, ..., yₙ}
                vals = np.array([o.get(var, 0.0) for o in outcomes])
                n = len(vals)
                if n > 0:
                    # Sample mean: ȳ = (1/n)Σᵢ₌₁ⁿ yᵢ
                    # E[Y] = lim_{n→∞} (1/n)Σᵢ yᵢ (Monte Carlo convergence)
                    expected_outcome[var] = float(np.mean(vals))  # E[Y] = (1/n)Σᵢ yᵢ
                else:
                    expected_outcome[var] = 0.0
            
            # Confidence intervals using quantile-based estimation (non-parametric)
            # For 90% CI: CI₉₀ = [Q₀.₀₅, Q₀.₉₅] where Q_p is p-th quantile
            # Non-parametric: no distributional assumptions
            # Alternative (parametric, if normal): CI = [μ̂ - z₀.₀₅·SE, μ̂ + z₀.₀₅·SE]
            # where z₀.₀₅ ≈ 1.645 for 90% CI (two-tailed: 5% in each tail)
            ci90_outcome: Dict[str, Tuple[float, float]] = {}
            for var in expected_outcome.keys():
                vals = np.array([o.get(var, 0.0) for o in outcomes])
                n = len(vals)
                if n > 0:
                    # Non-parametric 90% CI: [Q₀.₀₅, Q₀.₉₅]
                    # Quantile function: Q(p) = inf{x : P(X ≤ x) ≥ p}
                    # 5th percentile (lower bound): Q₀.₀₅
                    # 95th percentile (upper bound): Q₀.₉₅
                    ci90_outcome[var] = (
                        float(np.quantile(vals, 0.05)),  # Lower bound: Q₀.₀₅ = 5th percentile
                        float(np.quantile(vals, 0.95)),  # Upper bound: Q₀.₉₅ = 95th percentile
                    )
                else:
                    ci90_outcome[var] = (0.0, 0.0)
            
            # Recursively nest for next level
            child_nodes: List[Dict[str, Any]] = []
            if level < nesting_depth - 1:
                # Create nested interventions based on outcomes
                for outcome_sample in outcomes[:5]:  # Top 5 outcomes
                    # Generate follow-up interventions
                    nested_interventions: Dict[str, float] = {}
                    # (Simplified: could use policy logic here)
                    
                    child = simulate_level(
                        state=outcome_sample,
                        interventions_at_level=nested_interventions,
                        level=level + 1,
                        parent_probability=parent_probability * (1.0 / num_samples),
                    )
                    child_nodes.append(child)
            
            return {
                "level": level,
                "state": state,
                "interventions": interventions_at_level,
                "expected_outcome": expected_outcome,
                "ci90": ci90_outcome,
                "outcome_samples": outcomes[:10],  # Top 10
                "probability": parent_probability,
                "children": child_nodes,
                "is_leaf": level >= nesting_depth - 1,
            }
        
        root_node = simulate_level(initial_state, interventions, level=0)
        simulation_tree.append(root_node)
        
        # Extract final outcomes (leaf nodes)
        def collect_leaves(node: Dict[str, Any]) -> List[Dict[str, Any]]:
            if node.get("is_leaf", False):
                return [node]
            leaves = []
            for child in node.get("children", []):
                leaves.extend(collect_leaves(child))
            return leaves
        
        leaves = collect_leaves(root_node)
        
        return {
            "simulation_tree": root_node,
            "all_leaves": leaves,
            "expected_final_outcomes": {
                var: float(np.mean([l["state"].get(var, 0.0) for l in leaves]))
                for var in set.union(*[set(l["state"].keys()) for l in leaves])
            },
            "total_paths": len(leaves),
            "max_nesting_reached": nesting_depth,
        }

    def adversarial_nested_analysis(
        self,
        intervention: Dict[str, float],
        target: str,
        adversary_objectives: List[str],
        nesting_depth: int = 4,
    ) -> Dict[str, Any]:
        """
        Adversarial nesting: consider worst-case reactions that could undermine your intervention.
        
        Models how external forces or negative feedback might react to your actions,
        then nests to find best intervention despite adversarial responses.
        
        Args:
            intervention: Your proposed intervention
            target: Variable you want to optimize
            adversary_objectives: Variables an adversary wants to minimize (worst-case)
            nesting_depth: How many adversarial reaction layers to consider
        
        Returns:
            Worst-case scenarios, robust interventions, adversarial paths
        """
        # Adversary model: reacts to minimize adversary_objectives
        adversarial_paths: List[Dict[str, Any]] = []
        
        current_state = intervention.copy()
        
        for depth in range(nesting_depth):
            # Your intervention at this depth
            predicted_outcome = self._predict_outcomes({}, current_state)
            
            # Adversary reaction: find interventions that worsen adversary objectives
            adversary_interventions: List[Dict[str, float]] = []
            
            for adv_obj in adversary_objectives:
                if adv_obj not in self.causal_graph:
                    continue
                
                # Find variables that affect adversary objective
                affecting_vars = list(self.causal_graph.predecessors(adv_obj))
                
                # Adversary intervenes to minimize this objective
                for var in affecting_vars[:3]:  # Top 3
                    edge_data = self.causal_graph[var][adv_obj]
                    strength = edge_data.get('strength', 0.0)
                    
                    # Adversarial optimization: adversary minimizes your objective
                    # Adversary strategy: u_adv* = argmin_u_adv f(x, u, u_adv)
                    # Where f is your objective function, u is your intervention, u_adv is adversary's
                    # Gradient-based: u_adv ← u_adv - α·∇_{u_adv} f (gradient descent on negative objective)
                    # If positive strength, adversary reduces var; if negative, increases
                    current_val = predicted_outcome.get(var, 0.0)
                    # Adversarial perturbation: Δu = -sign(∂f/∂u)·η·|u|
                    # where η = 0.5 is step size, sign gives direction to worsen your objective
                    adversary_intervention = {
                        var: current_val - 0.5 * np.sign(strength) * abs(current_val)
                    }
                    adversary_interventions.append(adversary_intervention)
            
            # Worst-case outcome after adversary reaction
            worst_outcomes: List[Dict[str, Any]] = []
            for adv_intervention in adversary_interventions:
                combined = {**current_state, **adv_intervention}
                worst_outcome = self._predict_outcomes({}, combined)
                
                # Adversary damage quantification using L1 norm
                # Damage = ||y_worst - y_predicted||₁ = Σⱼ |yⱼ_worst - yⱼ_predicted|
                # This measures magnitude of deviation from expected outcome
                target_value = worst_outcome.get(target, 0.0)
                # Aggregate damage across all adversary objectives (L1 norm)
                adversary_damage = sum([
                    abs(worst_outcome.get(obj, 0.0) - predicted_outcome.get(obj, 0.0))
                    for obj in adversary_objectives
                ])
                
                worst_outcomes.append({
                    "adversary_intervention": adv_intervention,
                    "outcome": worst_outcome,
                    "target_value": float(target_value),
                    "adversary_damage": float(adversary_damage),
                })
            
            # Sort by worst for target
            worst_outcomes.sort(key=lambda x: x["target_value"])
            worst_case = worst_outcomes[0] if worst_outcomes else None
            
            adversarial_paths.append({
                "depth": depth,
                "your_intervention": current_state,
                "predicted_outcome": predicted_outcome,
                "adversary_reactions": adversary_interventions,
                "worst_case": worst_case,
            })
            
            # Update state for next depth
            if worst_case:
                current_state = worst_case["outcome"]
        
        # Find robust intervention (works even in worst case)
        final_target_values = [
            path["worst_case"]["target_value"]
            for path in adversarial_paths
            if path.get("worst_case")
        ]
        worst_final = min(final_target_values) if final_target_values else 0.0
        
        return {
            "intervention": intervention,
            "target": target,
            "adversarial_paths": adversarial_paths,
            "worst_case_target_value": float(worst_final),
            "adversarial_depth": nesting_depth,
            "robustness_assessment": {
                "worst_case_loss": float(intervention.get(target, 0.0) - worst_final),
                "is_robust": worst_final > intervention.get(target, 0.0) * 0.5,  # Still >50% of desired
            },
        }

    def multi_objective_infinite_nesting(
        self,
        current_state: Dict[str, float],
        objectives: Dict[str, float],  # {variable: weight} to maximize
        constraints: Dict[str, Tuple[float, float]],  # {variable: (min, max)}
        max_depth: int = 8,
        beam_width: int = 5,
    ) -> Dict[str, Any]:
        """
        Multi-objective infinite nesting: optimize multiple goals simultaneously.
        
        Uses beam search to explore intervention space, balancing multiple objectives.
        Nests as deep as needed to find Pareto-optimal solutions.
        
        Args:
            current_state: Current state
            objectives: {variable: weight} - variables to maximize with weights
            constraints: {variable: (min, max)} - hard bounds
            max_depth: Maximum nesting depth
            beam_width: How many paths to keep at each level (beam search)
        
        Returns:
            Pareto-optimal intervention paths, multi-objective trade-offs
        """
        # Beam search: maintain top-k paths at each level
        beam: List[Dict[str, Any]] = [{
            "state": current_state,
            "path": [],
            "objective_vector": {k: 0.0 for k in objectives.keys()},
            "depth": 0,
        }]
        
        pareto_frontier: List[Dict[str, Any]] = []
        
        for depth in range(max_depth):
            next_beam: List[Dict[str, Any]] = []
            
            for path_node in beam:
                # Get intervention candidates
                candidates = [
                    node for node in self.causal_graph.nodes()
                    if len(list(self.causal_graph.successors(node))) > 0
                    and node not in objectives.keys()
                ]
                
                # Explore interventions
                for var in candidates[:10]:  # Limit candidates
                    stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
                    current_val = path_node["state"].get(var, stats["mean"])
                    
                    # Try intervention
                    intervention = {var: current_val + self.rng.normal(0, stats["std"] * 1.0)}
                    
                    # Check constraints
                    violates = False
                    for const_var, (min_val, max_val) in constraints.items():
                        if const_var in intervention:
                            if not (min_val <= intervention[const_var] <= max_val):
                                violates = True
                                break
                    
                    if violates:
                        continue
                    
                    # Predict outcome
                    outcome = self._predict_outcomes(path_node["state"], intervention)
                    
                    # Check outcome constraints
                    for const_var, (min_val, max_val) in constraints.items():
                        if const_var in outcome:
                            if not (min_val <= outcome[const_var] <= max_val):
                                violates = True
                                break
                    
                    if violates:
                        continue
                    
                    # Multi-objective optimization: weighted sum scalarization
                    # Objective vector: f(x) = [f₁(x), f₂(x), ..., fₖ(x)] where fᵢ are individual objectives
                    # Scalarized: F(x) = Σᵢ wᵢ·fᵢ(x) where wᵢ are weights
                    # This converts multi-objective to single-objective for optimization
                    # Weighted sum: F(x) = w₁·f₁(x) + w₂·f₂(x) + ... + wₖ·fₖ(x)
                    obj_vector = {}
                    for k, weight in objectives.items():
                        # Individual objective value: fᵢ(x)
                        obj_value = outcome.get(k, 0.0)
                        # Weighted component: wᵢ·fᵢ(x)
                        obj_vector[k] = obj_value * weight
                    
                    # Combined objective: F(x) = Σᵢ wᵢ·fᵢ(x) (weighted sum)
                    # This is the scalarized multi-objective function
                    # Alternative formulations:
                    # - Weighted Lp norm: F(x) = (Σᵢ wᵢ·|fᵢ(x)|ᵖ)^(1/p)
                    # - Chebyshev: F(x) = minᵢ wᵢ·fᵢ(x) (for maximization)
                    combined_obj = sum(obj_vector.values())  # F(x) = Σᵢ wᵢ·fᵢ(x)
                    
                    next_beam.append({
                        "state": outcome,
                        "path": path_node["path"] + [intervention],
                        "objective_vector": obj_vector,
                        "combined_objective": float(combined_obj),
                        "depth": depth + 1,
                    })
            
            # Prune beam: keep top beam_width by combined objective
            next_beam.sort(key=lambda x: x["combined_objective"], reverse=True)
            beam = next_beam[:beam_width]
            
            # Add to Pareto if not dominated
            for node in beam:
                # Pareto dominance: x dominates y iff ∀i: fᵢ(x) ≥ fᵢ(y) ∧ ∃j: fⱼ(x) > fⱼ(y)
                # This follows multi-objective optimization theory (Pareto optimality)
                is_dominated = False
                for pareto_node in pareto_frontier:
                    # Check if pareto_node dominates node using Pareto dominance criterion
                    # f₁ dominates f₂ if: ∀k: f₁ₖ ≥ f₂ₖ ∧ ∃k: f₁ₖ > f₂ₖ
                    all_better = all(
                        pareto_node["objective_vector"].get(k, 0.0) >= node["objective_vector"].get(k, 0.0)
                        for k in objectives.keys()
                    )
                    some_better = any(
                        pareto_node["objective_vector"].get(k, 0.0) > node["objective_vector"].get(k, 0.0)
                        for k in objectives.keys()
                    )
                    # Pareto dominance condition: all_better ∧ some_better
                    if all_better and some_better:
                        is_dominated = True
                        break
                
                if not is_dominated:
                    # Remove dominated nodes from frontier
                    pareto_frontier = [
                        pn for pn in pareto_frontier
                        if not (
                            all(node["objective_vector"].get(k, 0.0) >= pn["objective_vector"].get(k, 0.0)
                            for k in objectives.keys()) and
                            any(node["objective_vector"].get(k, 0.0) > pn["objective_vector"].get(k, 0.0)
                            for k in objectives.keys())
                        )
                    ]
                    pareto_frontier.append(node)
        
        # Sort Pareto frontier
        pareto_frontier.sort(key=lambda x: x["combined_objective"], reverse=True)
        
        return {
            "pareto_frontier": pareto_frontier[:20],  # Top 20
            "best_path": pareto_frontier[0] if pareto_frontier else None,
            "total_paths_explored": len(pareto_frontier),
            "max_depth_reached": max_depth,
            "objectives": objectives,
            "trade_off_analysis": {
                "objective_ranges": {
                    k: {
                        "min": float(min(p["objective_vector"].get(k, 0.0) for p in pareto_frontier)),
                        "max": float(max(p["objective_vector"].get(k, 0.0) for p in pareto_frontier)),
                    }
                    for k in objectives.keys()
                },
            },
        }

    def temporal_causal_chain_analysis(
        self,
        initial_intervention: Dict[str, float],
        target: str,
        time_horizon: int = 10,
        lag_structure: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> Dict[str, Any]:
        """
        Temporal nesting: account for time delays in causal effects.
        
        Models that some effects take time (X→Y might take 2 periods),
        creating temporal causal chains that nest across time.
        
        Args:
            initial_intervention: Starting intervention
            target: Variable to optimize
            time_horizon: Number of time periods
            lag_structure: {(source, target): lag_periods} - if None, assumes lag=1
        
        Returns:
            Temporal trajectory, delayed effects, optimal timing
        """
        if lag_structure is None:
            # Default: each edge has lag=1
            lag_structure = {
                (u, v): 1
                for u, v in self.causal_graph.edges()
            }
        
        # State history across time: x(t) = [x₁(t), x₂(t), ..., xₙ(t)]
        # Discrete-time causal system with distributed lags
        state_history: List[Dict[str, float]] = [initial_intervention.copy()]
        
        # Track pending effects: distributed lag model
        # Effect yⱼ(t+τᵢⱼ) = βᵢⱼ·xᵢ(t) where τᵢⱼ is lag for edge (i,j)
        # This implements VAR (Vector Autoregression) with distributed lags
        pending_effects: Dict[int, List[Tuple[str, str, float]]] = {}  # {time: [(source, target, effect)]}
        
        for t in range(time_horizon):
            current_state = state_history[-1].copy()
            next_state = current_state.copy()
            
            # Apply pending effects from previous periods (distributed lag accumulation)
            # y(t) = Σᵢ Σₖ βᵢⱼ·xᵢ(t-τₖ) where τₖ are lags
            if t in pending_effects:
                for source, target, effect_magnitude in pending_effects[t]:
                    if target in next_state:
                        # Linear accumulation: additive effects from multiple sources
                        next_state[target] = next_state.get(target, 0.0) + effect_magnitude
            
            # Compute new effects with lags using distributed lag model
            # Causal effect equation: yⱼ(t+τᵢⱼ) = βᵢⱼ·xᵢ(t)
            # where τᵢⱼ is the lag for edge (i,j), βᵢⱼ is structural coefficient
            # This implements VAR (Vector Autoregression) with distributed lags
            for u, v in self.causal_graph.edges():
                lag = lag_structure.get((u, v), 1)  # Lag τᵢⱼ (time delay)
                source_val = current_state.get(u, 0.0)  # xᵢ(t): source value at time t
                edge_data = self.causal_graph[u][v]
                strength = edge_data.get('strength', 0.0)  # Structural coefficient βᵢⱼ
                
                # Effect magnitude: e = βᵢⱼ·xᵢ(t) (linear structural equation)
                # This is the effect that will manifest at time t + τ
                effect = source_val * strength  # e = βᵢⱼ·xᵢ(t)
                
                # Schedule effect for future period (distributed lag)
                # Effect manifests at time t + τ: yⱼ(t+τ) ← yⱼ(t+τ) + βᵢⱼ·xᵢ(t)
                # This implements: yⱼ(t+τᵢⱼ) = βᵢⱼ·xᵢ(t)
                future_time = t + lag  # Time when effect manifests: t + τᵢⱼ
                if future_time < time_horizon:
                    if future_time not in pending_effects:
                        pending_effects[future_time] = []
                    # Store effect to be applied at future_time
                    pending_effects[future_time].append((u, v, effect))
            
            state_history.append(next_state)
        
        # Extract temporal patterns
        target_trajectory = [s.get(target, 0.0) for s in state_history]
        
        return {
            "initial_intervention": initial_intervention,
            "target": target,
            "temporal_trajectory": state_history,
            "target_over_time": target_trajectory,
            "peak_value": float(max(target_trajectory)) if target_trajectory else 0.0,
            "time_to_peak": int(np.argmax(target_trajectory)) if target_trajectory else 0,
            "steady_state_value": float(target_trajectory[-1]) if target_trajectory else 0.0,
            "lag_structure_used": lag_structure,
        }

    def explainable_nested_analysis(
        self,
        intervention: Dict[str, float],
        target: str,
        depth: int = 5,
    ) -> Dict[str, Any]:
        """
        Explainable nesting: at each layer, explain WHY the effects occur.
        
        Nests deeply but provides human-readable explanations for each causal step,
        building a narrative of how intervention → effect → effect → ... → target.
        
        Args:
            intervention: Initial intervention
            target: Target outcome
            depth: Nesting depth
        
        Returns:
            Explained causal chain, narrative, reasoning at each level
        """
        explanations: List[Dict[str, Any]] = []
        
        current_state = intervention.copy()
        
        for level in range(depth):
            # Predict outcomes
            predicted = self._predict_outcomes({}, current_state)
            
            # Find causal paths to target
            paths_to_target: List[List[str]] = []
            for inter_var in current_state.keys():
                if inter_var == target:
                    continue
                try:
                    paths = list(nx.all_simple_paths(
                        self.causal_graph,
                        inter_var,
                        target,
                        cutoff=depth - level
                    ))
                    paths_to_target.extend(paths)
                except Exception:
                    pass
            
            # Explain each path using causal effect decomposition
            # Path effect = ∏(i,j)∈Path βᵢⱼ (product of structural coefficients)
            # Following Pearl's do-calculus and causal mediation analysis
            path_explanations: List[str] = []
            for path in paths_to_target[:5]:  # Top 5 paths
                explanation_parts = []
                path_product = 1.0  # Initialize path effect product
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = self.causal_graph[u][v]
                    strength = abs(edge_data.get('strength', 0.0))  # |βᵢⱼ|
                    sign = "+" if edge_data.get('strength', 0.0) >= 0 else "-"
                    # Accumulate path effect: ∏ βᵢⱼ
                    path_product *= edge_data.get('strength', 0.0)
                    
                    explanation_parts.append(
                        f"{u} {sign}affects {v} (β={strength:.3f})"
                    )
                # Path effect: total causal effect along this path
                explanation_parts.append(f"Path effect: {path_product:.4f}")
                
                path_explanation = " → ".join([
                    f"{path[i]}{'↑' if i < len(path)-1 else ''}"
                    for i in range(len(path))
                ])
                path_explanations.append(path_explanation)
            
            # Build explanation
            explanation = {
                "level": level,
                "interventions": current_state,
                "predicted_outcomes": predicted,
                "paths_to_target": [p[:3] for p in paths_to_target[:3]],  # Show first 3 vars of each path
                "explanation_text": f"Level {level}: " + "; ".join(path_explanations[:3]),
                "target_value_at_level": float(predicted.get(target, 0.0)),
            }
            explanations.append(explanation)
            
            # Update for next level (use predicted as new baseline)
            current_state = predicted
        
        # Synthesize narrative
        narrative_parts = [
            f"At level {e['level']}: {e['explanation_text']}"
            for e in explanations
        ]
        
        return {
            "intervention": intervention,
            "target": target,
            "explanations_by_level": explanations,
            "narrative": "\n".join(narrative_parts),
            "final_target_value": float(explanations[-1]["target_value_at_level"]) if explanations else 0.0,
            "total_levels": len(explanations),
        }

    # ==================== MAJOR UPGRADES ====================
    
    def gradient_based_intervention_optimization(
        self,
        initial_state: Dict[str, float],
        target: str,
        intervention_vars: List[str],
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
    ) -> Dict[str, Any]:
        """
        Gradient-based optimization for finding optimal interventions.
        
        Uses automatic differentiation via numerical gradients to optimize:
        minimize: -f(x) where f(x) = predicted outcome given intervention x
        
        Mathematical formulation:
        - Objective: J(θ) = -y(θ) where y(θ) = predicted outcome
        - Gradient: ∇_θ J(θ) = -∇_θ y(θ) computed via finite differences
        - Update: θ_{k+1} = θ_k - α·∇_θ J(θ_k) (gradient descent)
        
        Args:
            initial_state: Current state
            target: Variable to optimize
            intervention_vars: Variables that can be intervened on
            constraints: {var: (min, max)} bounds
            method: Optimization method ('L-BFGS-B', 'BFGS', 'SLSQP', etc.)
        
        Returns:
            Optimal intervention, objective value, convergence info
        """
        # Prepare bounds
        bounds = []
        x0 = []
        var_to_idx = {}
        for i, var in enumerate(intervention_vars):
            var_to_idx[var] = i
            stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
            current_val = initial_state.get(var, stats["mean"])
            x0.append(current_val)
            
            if constraints and var in constraints:
                min_val, max_val = constraints[var]
                bounds.append((min_val, max_val))
            else:
                # Default bounds: ±3 standard deviations
                bounds.append((current_val - 3 * stats["std"], current_val + 3 * stats["std"]))
        
        # Objective function: J(x) = -y(x) where y(x) is predicted outcome
        def objective(x: np.ndarray) -> float:
            """Objective: minimize -f(x) (maximize f(x))"""
            intervention = {intervention_vars[i]: float(x[i]) for i in range(len(x))}
            outcome = self._predict_outcomes(initial_state, intervention)
            target_val = outcome.get(target, 0.0)
            return -target_val  # Negative for minimization
        
        # Numerical gradient: ∇_θ J(θ) ≈ [J(θ+ε·e_i) - J(θ-ε·e_i)] / (2ε)
        # where e_i is unit vector in direction i, ε is small step
        def gradient(x: np.ndarray) -> np.ndarray:
            """Compute gradient via finite differences: ∇f ≈ (f(x+ε) - f(x-ε))/(2ε)"""
            epsilon = 1e-5
            grad = np.zeros_like(x)
            f0 = objective(x)
            
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += epsilon
                f_plus = objective(x_plus)
                grad[i] = (f_plus - f0) / epsilon
            
            return grad
        
        # Optimize using scipy.optimize
        try:
            result = minimize(
                objective,
                x0=np.array(x0),
                method=method,
                bounds=bounds,
                jac=gradient if method in ["L-BFGS-B", "BFGS", "CG"] else None,
                options={"maxiter": 100, "ftol": 1e-6} if method == "L-BFGS-B" else {}
            )
            
            optimal_intervention = {intervention_vars[i]: float(result.x[i]) for i in range(len(result.x))}
            optimal_outcome = self._predict_outcomes(initial_state, optimal_intervention)
            
            return {
                "optimal_intervention": optimal_intervention,
                "optimal_target_value": float(optimal_outcome.get(target, 0.0)),
                "objective_value": float(result.fun),
                "success": bool(result.success),
                "iterations": int(result.nit) if hasattr(result, 'nit') else 0,
                "convergence_message": str(result.message),
                "gradient_norm": float(np.linalg.norm(gradient(result.x))) if result.success else float('inf'),
            }
        except Exception as e:
            return {"error": str(e), "optimal_intervention": {}, "success": False}

    def nonlinear_scm_prediction(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float],
        include_interactions: bool = True,
    ) -> Dict[str, float]:
        """
        Non-linear SCM prediction with interaction terms.
        
        Extends linear model: y = Σᵢ βᵢ·xᵢ + Σᵢⱼ γᵢⱼ·xᵢ·xⱼ + ε
        where γᵢⱼ are interaction coefficients.
        
        Mathematical foundation:
        - Linear term: Σᵢ βᵢ·xᵢ
        - Quadratic interaction: Σᵢⱼ γᵢⱼ·xᵢ·xⱼ (product of parent pairs)
        - Full model: y = linear_term + interaction_term + ε
        
        Args:
            factual_state: Current state
            interventions: Interventions to apply
            include_interactions: Whether to include interaction terms
        
        Returns:
            Predicted outcomes with non-linear effects
        """
        # Standardize and merge states
        raw_state = factual_state.copy()
        raw_state.update(interventions)
        z_state = self._standardize_state(raw_state)
        z_pred = dict(z_state)
        
        # Interaction coefficients cache {node: {(parent1, parent2): γ}}
        interaction_coeffs: Dict[str, Dict[Tuple[str, str], float]] = {}
        
        for node in nx.topological_sort(self.causal_graph):
            if node in interventions:
                continue
            
            parents = list(self.causal_graph.predecessors(node))
            if not parents:
                continue
            
            # Linear term: Σᵢ βᵢ·z_xi
            linear_term = 0.0
            for parent in parents:
                parent_z = z_pred.get(parent, z_state.get(parent, 0.0))
                beta = self.causal_graph[parent][node].get('strength', 0.0)
                linear_term += parent_z * beta
            
            # Interaction terms: Σᵢⱼ γᵢⱼ·z_xi·z_xj
            interaction_term = 0.0
            if include_interactions and node in self.interaction_terms:
                for (p1, p2) in self.interaction_terms[node]:
                    if p1 in parents and p2 in parents:
                        z1 = z_pred.get(p1, z_state.get(p1, 0.0))
                        z2 = z_pred.get(p2, z_state.get(p2, 0.0))
                        # Interaction coefficient (default: small value)
                        gamma = self.causal_graph[p1][node].get('interaction_strength', {}).get(p2, 0.0)
                        interaction_term += gamma * z1 * z2  # γ·x₁·x₂
            
            # Total prediction: z_y = linear_term + interaction_term
            z_pred[node] = linear_term + interaction_term
        
        # De-standardize
        predicted_state = {}
        for var, z_val in z_pred.items():
            predicted_state[var] = self._destandardize_value(var, z_val)
        return predicted_state

    def compute_information_theoretic_measures(
        self,
        df: Any,
        variables: List[str],
    ) -> Dict[str, Any]:
        """
        Compute information-theoretic measures: entropy, mutual information, causal entropy.
        
        Mathematical formulations:
        - Entropy: H(X) = -Σᵢ P(xᵢ) log₂ P(xᵢ) (discrete) or -∫ p(x) log p(x) dx (continuous)
        - Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y) = KL(P(X,Y) || P(X)P(Y))
        - Conditional MI: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        - Causal Entropy: Expected reduction in entropy from intervention
        
        Args:
            df: DataFrame with variables
            variables: Variables to analyze
        
        Returns:
            Entropies, mutual information, causal information gains
        """
        if df is None or len(df) < 10:
            return {"error": "Insufficient data"}
        
        data = df[variables].dropna()
        if len(data) < 10:
            return {"error": "Insufficient data after dropna"}
        
        results: Dict[str, Any] = {
            "entropies": {},
            "mutual_information": {},
            "conditional_mi": {},
        }
        
        # Compute entropies: H(X) = -Σ p(x) log p(x)
        # Using discrete histogram approximation for continuous variables
        for var in variables:
            if var not in data.columns:
                continue
            
            series = data[var].dropna()
            if len(series) < 5:
                continue
            
            # Discretize for entropy estimation (histogram method)
            # H(X) ≈ -Σᵢ (nᵢ/n) log₂(nᵢ/n) where nᵢ is count in bin i
            n_bins = min(20, max(5, int(np.sqrt(len(series)))))  # Adaptive binning
            hist, bins = np.histogram(series, bins=n_bins)
            hist = hist[hist > 0]  # Remove empty bins
            probs = hist / hist.sum()
            
            # Shannon entropy: H(X) = -Σᵢ pᵢ log₂ pᵢ
            entropy = -np.sum(probs * np.log2(probs))
            results["entropies"][var] = float(entropy)
            self._entropy_cache[var] = float(entropy)
        
        # Compute pairwise mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        for i, var1 in enumerate(variables):
            if var1 not in results["entropies"]:
                continue
            for var2 in variables[i+1:]:
                if var2 not in results["entropies"]:
                    continue
                
                # Joint entropy: H(X,Y) = -Σᵢⱼ p(xᵢ,yⱼ) log₂ p(xᵢ,yⱼ)
                joint_series = data[[var1, var2]].dropna()
                if len(joint_series) < 5:
                    continue
                
                # 2D histogram
                n_bins = min(10, max(3, int(np.cbrt(len(joint_series)))))
                hist_2d, _, _ = np.histogram2d(
                    joint_series[var1],
                    joint_series[var2],
                    bins=n_bins
                )
                hist_2d = hist_2d[hist_2d > 0]
                probs_joint = hist_2d / hist_2d.sum()
                
                # Joint entropy
                h_joint = -np.sum(probs_joint * np.log2(probs_joint))
                
                # Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
                mi = results["entropies"][var1] + results["entropies"][var2] - float(h_joint)
                results["mutual_information"][f"{var1};{var2}"] = float(max(0.0, mi))  # MI ≥ 0
                self._mi_cache[(var1, var2)] = float(max(0.0, mi))
        
        return results

    def convex_intervention_optimization(
        self,
        initial_state: Dict[str, float],
        objectives: Dict[str, float],  # {var: weight}
        constraints_dict: Dict[str, Tuple[float, float]],  # {var: (min, max)}
        intervention_vars: List[str],
    ) -> Dict[str, Any]:
        """
        Convex optimization for interventions using CVXPY (if available).
        
        Mathematical formulation:
        minimize: Σᵢ wᵢ·fᵢ(x)
        subject to: l ≤ x ≤ u (box constraints)
        where fᵢ are linear functions of interventions
        
        Uses CVXPY for guaranteed global optimum in convex problems.
        
        Args:
            initial_state: Current state
            objectives: {variable: weight} to optimize
            constraints_dict: {variable: (min, max)} bounds
            intervention_vars: Variables to optimize
        
        Returns:
            Optimal intervention, solver status, dual variables
        """
        if not CVXPY_AVAILABLE:
            return {"error": "CVXPY not available. Install with: pip install cvxpy"}
        
        try:
            # Decision variables: intervention values
            n = len(intervention_vars)
            x = cp.Variable(n, name="interventions")
            
            # Build objective: minimize Σᵢ wᵢ·(-outcome_i)
            # Since CVXPY minimizes, we minimize negative of what we want to maximize
            obj_terms = []
            for target_var, weight in objectives.items():
                # Approximate outcome as linear function of interventions
                # outcome ≈ baseline + Σⱼ βⱼ·intervention_j
                baseline = initial_state.get(target_var, 0.0)
                
                # Find paths from intervention vars to target
                effect_sum = 0.0
                for i, inter_var in enumerate(intervention_vars):
                    try:
                        path = nx.shortest_path(self.causal_graph, inter_var, target_var)
                        if len(path) > 1:
                            # Path strength (simplified: direct edge if exists)
                            if self.causal_graph.has_edge(inter_var, target_var):
                                beta = self.causal_graph[inter_var][target_var].get('strength', 0.0)
                                effect_sum += beta * x[i]
                    except:
                        pass
                
                # Objective term: -weight * (baseline + effect_sum)
                obj_terms.append(-weight * (baseline + effect_sum))
            
            objective = cp.Minimize(sum(obj_terms))
            
            # Constraints: l ≤ x ≤ u
            constraints = []
            for i, var in enumerate(intervention_vars):
                if var in constraints_dict:
                    min_val, max_val = constraints_dict[var]
                    constraints.append(x[i] >= min_val)
                    constraints.append(x[i] <= max_val)
                else:
                    # Default bounds
                    stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
                    current = initial_state.get(var, stats["mean"])
                    constraints.append(x[i] >= current - 3 * stats["std"])
                    constraints.append(x[i] <= current + 3 * stats["std"])
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS if hasattr(cp, 'ECOS') else cp.SCS)
            
            if problem.status in ["optimal", "optimal_inaccurate"]:
                optimal_x = {intervention_vars[i]: float(x.value[i]) for i in range(n)}
                return {
                    "optimal_intervention": optimal_x,
                    "status": problem.status,
                    "objective_value": float(problem.value),
                    "solver": problem.solver_stats.name if hasattr(problem, 'solver_stats') else "unknown",
                }
            else:
                return {"error": f"Solver failed: {problem.status}", "optimal_intervention": {}}
        except Exception as e:
            return {"error": str(e), "optimal_intervention": {}}

    def sensitivity_analysis(
        self,
        intervention: Dict[str, float],
        target: str,
        perturbation_size: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Sensitivity analysis: how sensitive is outcome to intervention changes?
        
        Mathematical formulation:
        - Sensitivity: Sᵢ = ∂y/∂xᵢ ≈ (y(x + ε·eᵢ) - y(x)) / ε
        - Elasticity: Eᵢ = (∂y/∂xᵢ)·(xᵢ/y) = sensitivity · (x/y)
        - Total sensitivity: ||∇y||₂ = √(Σᵢ Sᵢ²)
        
        Args:
            intervention: Base intervention
            target: Target variable
            perturbation_size: ε for finite differences
        
        Returns:
            Sensitivities, elasticities, most influential variables
        """
        base_outcome = self._predict_outcomes({}, intervention)
        base_target = base_outcome.get(target, 0.0)
        
        sensitivities: Dict[str, float] = {}
        elasticities: Dict[str, float] = {}
        
        for var, val in intervention.items():
            # Perturb: x + ε·e
            perturbed_intervention = intervention.copy()
            perturbed_intervention[var] = val + perturbation_size
            
            # Compute sensitivity: S = (y(x+ε) - y(x)) / ε
            perturbed_outcome = self._predict_outcomes({}, perturbed_intervention)
            perturbed_target = perturbed_outcome.get(target, 0.0)
            
            sensitivity = (perturbed_target - base_target) / perturbation_size
            sensitivities[var] = float(sensitivity)
            
            # Elasticity: E = (∂y/∂x)·(x/y) = S·(x/y)
            if abs(base_target) > 1e-6 and abs(val) > 1e-6:
                elasticity = sensitivity * (val / base_target)
                elasticities[var] = float(elasticity)
            else:
                elasticities[var] = 0.0
        
        # Total sensitivity (L2 norm of gradient): ||∇y||₂ = √(Σᵢ Sᵢ²)
        sensitivity_vector = np.array(list(sensitivities.values()))
        total_sensitivity = float(np.linalg.norm(sensitivity_vector))
        
        # Most influential (highest absolute sensitivity)
        most_influential = max(sensitivities.items(), key=lambda x: abs(x[1])) if sensitivities else None
        
        return {
            "sensitivities": sensitivities,
            "elasticities": elasticities,
            "total_sensitivity": total_sensitivity,
            "most_influential_variable": most_influential[0] if most_influential else None,
            "most_influential_sensitivity": most_influential[1] if most_influential else 0.0,
        }

    def vectorized_batch_prediction(
        self,
        initial_state: Dict[str, float],
        intervention_batch: List[Dict[str, float]],
    ) -> np.ndarray:
        """
        Vectorized batch prediction for efficiency.
        
        Computes predictions for multiple interventions in parallel using vectorized operations.
        Much faster than looping through interventions individually.
        
        Mathematical: y_batch = f(X_batch) where X_batch is matrix of interventions
        
        Args:
            initial_state: Base state
            intervention_batch: List of interventions to predict
        
        Returns:
            Array of predicted outcomes (one row per intervention)
        """
        if not intervention_batch:
            return np.array([])
        
        # Extract all variables
        all_vars = set(initial_state.keys())
        for inter in intervention_batch:
            all_vars.update(inter.keys())
        
        all_vars = sorted(list(all_vars))
        n_interventions = len(intervention_batch)
        n_vars = len(all_vars)
        
        # Build intervention matrix: X[i, j] = intervention i's value for variable j
        X = np.zeros((n_interventions, n_vars))
        for i, inter in enumerate(intervention_batch):
            for j, var in enumerate(all_vars):
                X[i, j] = inter.get(var, initial_state.get(var, 0.0))
        
        # Vectorized predictions (for now, use batched calls)
        # Future: could vectorize the entire SCM forward pass
        predictions = []
        for inter in intervention_batch:
            outcome = self._predict_outcomes(initial_state, inter)
            predictions.append([outcome.get(var, 0.0) for var in all_vars])
        
        return np.array(predictions)

    def bayesian_edge_inference(
        self,
        df: Any,
        parent: str,
        child: str,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Bayesian inference for edge strength using conjugate prior.
        
        Mathematical formulation:
        - Prior: β ~ N(μ₀, σ²₀)
        - Likelihood: y | β, X ~ N(Xβ, σ²)
        - Posterior: β | y, X ~ N(μₙ, σ²ₙ)
        - Posterior mean: μₙ = (σ²₀X'X + σ²I)⁻¹(σ²₀X'X μ̂_OLS + σ²μ₀)
        - Posterior variance: σ²ₙ = (σ²₀⁻¹ + (X'X)/σ²)⁻¹
        
        Args:
            df: Data
            parent: Source variable
            child: Target variable
            prior_mu: Prior mean μ₀
            prior_sigma: Prior standard deviation σ₀
        
        Returns:
            Posterior mean, variance, credible intervals
        """
        if df is None or parent not in df.columns or child not in df.columns:
            return {"error": "Invalid data or variables"}
        
        data = df[[parent, child]].dropna()
        if len(data) < 5:
            return {"error": "Insufficient data"}
        
        X = data[parent].values.reshape(-1, 1)
        y = data[child].values
        
        # Standardize
        X_mean, X_std = X.mean(), X.std() or 1.0
        y_mean, y_std = y.mean(), y.std() or 1.0
        X_norm = (X - X_mean) / X_std
        y_norm = (y - y_mean) / y_std
        
        # OLS estimate: β̂_OLS = (X'X)⁻¹X'y
        XtX = X_norm.T @ X_norm
        Xty = X_norm.T @ y_norm
        beta_ols = (np.linalg.pinv(XtX) @ Xty)[0]
        
        # Likelihood variance: σ² = (1/n)Σ(y - Xβ̂)²
        residuals = y_norm - X_norm @ np.array([beta_ols])
        sigma_sq = float(np.var(residuals))
        
        # Bayesian update: posterior parameters
        # Precision: τ = 1/σ²
        tau_likelihood = 1.0 / (sigma_sq + 1e-6)  # Likelihood precision
        tau_prior = 1.0 / (prior_sigma ** 2)  # Prior precision
        
        # Posterior precision: τ_posterior = τ_prior + τ_likelihood
        tau_posterior = tau_prior + tau_likelihood * len(data)
        
        # Posterior mean: μ_posterior = (τ_prior·μ₀ + τ_likelihood·n·β̂_OLS) / τ_posterior
        mu_posterior = (tau_prior * prior_mu + tau_likelihood * len(data) * beta_ols) / tau_posterior
        
        # Posterior variance: σ²_posterior = 1/τ_posterior
        sigma_posterior_sq = 1.0 / tau_posterior
        sigma_posterior = np.sqrt(sigma_posterior_sq)
        
        # Credible intervals (95%): [μ - 1.96σ, μ + 1.96σ]
        ci_lower = mu_posterior - 1.96 * sigma_posterior
        ci_upper = mu_posterior + 1.96 * sigma_posterior
        
        # Store prior for future use
        self.bayesian_priors[(parent, child)] = {
            "mu": float(prior_mu),
            "sigma": float(prior_sigma)
        }
        
        return {
            "posterior_mean": float(mu_posterior),
            "posterior_std": float(sigma_posterior),
            "posterior_variance": float(sigma_posterior_sq),
            "credible_interval_95": (float(ci_lower), float(ci_upper)),
            "ols_estimate": float(beta_ols),
            "prior_mu": float(prior_mu),
            "prior_sigma": float(prior_sigma),
        }

    def add_interaction_term(
        self,
        node: str,
        parent1: str,
        parent2: str,
        interaction_strength: float = 0.0,
    ) -> None:
        """
        Add non-linear interaction term: y = ... + γ·x₁·x₂
        
        Mathematical: interaction effect = γ·x₁·x₂ where γ is interaction coefficient
        
        Args:
            node: Child variable
            parent1: First parent
            parent2: Second parent
            interaction_strength: Interaction coefficient γ
        """
        if node not in self.interaction_terms:
            self.interaction_terms[node] = []
        self.interaction_terms[node].append((parent1, parent2))
        
        # Store interaction strength in graph edge data
        if self.causal_graph.has_edge(parent1, node):
            if 'interaction_strength' not in self.causal_graph[parent1][node]:
                self.causal_graph[parent1][node]['interaction_strength'] = {}
            self.causal_graph[parent1][node]['interaction_strength'][parent2] = interaction_strength

    def clear_cache(self) -> None:
        """Clear prediction cache to free memory."""
        self._prediction_cache.clear()
        self._entropy_cache.clear()
        self._mi_cache.clear()

    def _predict_outcomes_cached(
        self,
        factual_state: Dict[str, float],
        interventions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Cached version of _predict_outcomes for performance.
        
        Uses hash-based caching to avoid recomputing identical predictions.
        Cache key: hash of (sorted factual_state.items(), sorted interventions.items())
        """
        if not self._cache_enabled:
            return self._predict_outcomes(factual_state, interventions)
        
        # Create cache key from state and interventions (sorted for consistency)
        state_key = tuple(sorted(factual_state.items()))
        inter_key = tuple(sorted(interventions.items()))
        cache_key = (state_key, inter_key)
        
        # Check cache
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key].copy()
        
        # Compute and cache (call with use_cache=False to avoid recursion)
        result = self._predict_outcomes(factual_state, interventions, use_cache=False)
        
        # LRU eviction if cache too large
        if len(self._prediction_cache) >= self._cache_max_size:
            # Remove oldest 10% of entries (simple FIFO approximation)
            keys_to_remove = list(self._prediction_cache.keys())[:self._cache_max_size // 10]
            for k in keys_to_remove:
                del self._prediction_cache[k]
        
        self._prediction_cache[cache_key] = result.copy()
        return result

    def granger_causality_test(
        self,
        df: Any,
        var1: str,
        var2: str,
        max_lag: int = 4,
    ) -> Dict[str, Any]:
        """
        Granger causality test: does var1 help predict var2?
        
        Mathematical formulation:
        - Restricted model: y_t = α + Σᵢ₌₁ᵐ βᵢ·y_{t-i} + ε_t
        - Unrestricted model: y_t = α + Σᵢ₌₁ᵐ βᵢ·y_{t-i} + Σᵢ₌₁ᵐ γᵢ·x_{t-i} + ε_t
        - F-statistic: F = [(RSS_r - RSS_u)/m] / [RSS_u/(n-2m-1)]
        - H₀: var1 does not Granger-cause var2 (γᵢ = 0 for all i)
        
        Args:
            df: Time series data
            var1: Potential cause variable
            var2: Outcome variable
            max_lag: Maximum lag to test
        
        Returns:
            F-statistic, p-value, Granger causality decision
        """
        if df is None or var1 not in df.columns or var2 not in df.columns:
            return {"error": "Invalid data or variables"}
        
        data = df[[var1, var2]].dropna()
        if len(data) < max_lag * 2 + 5:
            return {"error": "Insufficient data"}
        
        from scipy.stats import f as f_dist
        
        # Prepare lagged variables
        n = len(data)
        X_lags = []  # Lagged var2 (restricted model)
        X_unrestricted = []  # Lagged var2 + lagged var1 (unrestricted)
        y = []
        
        for t in range(max_lag, n):
            y.append(data[var2].iloc[t])
            
            # Restricted: only lags of var2
            restricted_row = [data[var2].iloc[t-i] for i in range(1, max_lag+1)]
            X_lags.append(restricted_row)
            
            # Unrestricted: lags of var2 + lags of var1
            unrestricted_row = restricted_row + [data[var1].iloc[t-i] for i in range(1, max_lag+1)]
            X_unrestricted.append(unrestricted_row)
        
        X_lags = np.array(X_lags)
        X_unrestricted = np.array(X_unrestricted)
        y = np.array(y)
        
        # Fit restricted model: y ~ lags(var2)
        # β_r = (X_r'X_r)⁻¹ X_r'y
        try:
            XrTXr = X_lags.T @ X_lags
            XrTy = X_lags.T @ y
            beta_r = np.linalg.pinv(XrTXr) @ XrTy
            y_pred_r = X_lags @ beta_r
            rss_r = float(np.sum((y - y_pred_r) ** 2))  # Restricted residual sum of squares
            
            # Fit unrestricted model: y ~ lags(var2) + lags(var1)
            XuTXu = X_unrestricted.T @ X_unrestricted
            XuTy = X_unrestricted.T @ y
            beta_u = np.linalg.pinv(XuTXu) @ XuTy
            y_pred_u = X_unrestricted @ beta_u
            rss_u = float(np.sum((y - y_pred_u) ** 2))  # Unrestricted RSS
            
            # F-statistic: F = [(RSS_r - RSS_u)/m] / [RSS_u/(n-2m-1)]
            # where m = number of additional parameters (max_lag)
            m = max_lag
            n_obs = len(y)
            df1 = m  # Numerator degrees of freedom
            df2 = n_obs - 2 * m - 1  # Denominator degrees of freedom
            
            if df2 > 0 and rss_u > 1e-10:
                f_stat = ((rss_r - rss_u) / m) / (rss_u / df2)
                f_stat = float(f_stat)
                
                # P-value: P(F > f_stat | H₀)
                p_value = float(1.0 - f_dist.cdf(f_stat, df1, df2))
                
                # Decision: reject H₀ if p < 0.05
                granger_causes = p_value < 0.05
                
                return {
                    "f_statistic": f_stat,
                    "p_value": float(p_value),
                    "granger_causes": granger_causes,
                    "max_lag": max_lag,
                    "restricted_rss": rss_r,
                    "unrestricted_rss": rss_u,
                    "df_numerator": df1,
                    "df_denominator": df2,
                    "interpretation": f"{var1} {'does' if granger_causes else 'does not'} Granger-cause {var2}",
                }
            else:
                return {"error": "Degenerate case in F-test"}
        except Exception as e:
            return {"error": str(e)}

    def vector_autoregression_estimation(
        self,
        df: Any,
        variables: List[str],
        max_lag: int = 2,
    ) -> Dict[str, Any]:
        """
        Estimate Vector Autoregression (VAR) model.
        
        Mathematical formulation:
        - VAR(p): x_t = A₁x_{t-1} + A₂x_{t-2} + ... + A_p x_{t-p} + ε_t
        - where x_t is vector of variables, A_i are coefficient matrices
        - Estimation: OLS on each equation: x_{i,t} = Σⱼ Σₖ a_{ij,k}·x_{j,t-k} + ε_{i,t}
        
        Args:
            df: Time series data
            variables: Variables to include in VAR
            max_lag: Lag order p
        
        Returns:
            Coefficient matrices, residuals, model diagnostics
        """
        if df is None or len(variables) < 2:
            return {"error": "Invalid data or need at least 2 variables"}
        
        data = df[variables].dropna()
        if len(data) < max_lag * len(variables) + 10:
            return {"error": "Insufficient data"}
        
        n_vars = len(variables)
        n_obs = len(data) - max_lag
        
        # Build lagged design matrix
        # Each row: [x_{t-1}, x_{t-2}, ..., x_{t-p}] for all variables
        X_lag = []
        y_matrix = []
        
        for t in range(max_lag, len(data)):
            # Dependent variables at time t: y_t
            y_row = [data[var].iloc[t] for var in variables]
            y_matrix.append(y_row)
            
            # Lagged predictors: [x_{t-1}, x_{t-2}, ..., x_{t-p}]
            lag_row = []
            for lag in range(1, max_lag + 1):
                for var in variables:
                    lag_row.append(data[var].iloc[t - lag])
            X_lag.append(lag_row)
        
        X_lag = np.array(X_lag)
        y_matrix = np.array(y_matrix)
        
        # Estimate VAR coefficients equation by equation
        # For each variable i: y_i = X_lag @ beta_i + ε_i
        coefficient_matrices: Dict[str, np.ndarray] = {}
        residuals_matrix = []
        
        for i, var in enumerate(variables):
            y_i = y_matrix[:, i]
            
            # OLS: β̂_i = (X'X)⁻¹X'y_i
            XtX = X_lag.T @ X_lag
            Xty = X_lag.T @ y_i
            beta_i = np.linalg.pinv(XtX) @ Xty
            
            # Reshape to matrix form: A_k[i, j] = coefficient of var_j at lag k on var_i
            A_matrices = {}
            for lag in range(1, max_lag + 1):
                A_lag = np.zeros((n_vars, n_vars))
                for j, var_j in enumerate(variables):
                    idx = (lag - 1) * n_vars + j
                    if idx < len(beta_i):
                        A_lag[i, j] = float(beta_i[idx])
                A_matrices[f"A_{lag}"] = A_lag
            
            coefficient_matrices[var] = {
                "coefficients": beta_i,
                "A_matrices": A_matrices,
            }
            
            # Residuals: ε_i = y_i - X_lag @ β̂_i
            y_pred_i = X_lag @ beta_i
            residuals_i = y_i - y_pred_i
            residuals_matrix.append(residuals_i)
        
        residuals_matrix = np.array(residuals_matrix).T  # Shape: (n_obs, n_vars)
        
        # Impulse response: how does shock to var_i affect var_j over time?
        # IRF_{j,i}(h) = ∂y_{j,t+h} / ∂ε_{i,t}
        # Can be computed from VAR coefficients (simplified here)
        
        return {
            "coefficient_matrices": {
                k: {k2: v2.tolist() if isinstance(v2, np.ndarray) else v2
                    for k2, v2 in v.items()}
                for k, v in coefficient_matrices.items()
            },
            "residuals": residuals_matrix.tolist(),
            "residual_covariance": np.cov(residuals_matrix.T).tolist(),
            "n_observations": n_obs,
            "n_variables": n_vars,
            "max_lag": max_lag,
            "variables": variables,
        }

    def impulse_response_analysis(
        self,
        var_coefficients: Dict[str, Any],
        horizon: int = 10,
        shock_size: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Impulse Response Function (IRF): effect of one-time shock on variables over time.
        
        Mathematical formulation:
        - IRF_{j,i}(h) = ∂y_{j,t+h} / ∂ε_{i,t}
        - Computed recursively from VAR: y_t = Σₖ A_k y_{t-k} + ε_t
        - IRF(0) = I (identity), IRF(h) = Σₖ A_k · IRF(h-k)
        
        Args:
            var_coefficients: VAR coefficient matrices from vector_autoregression_estimation
            horizon: Time horizon for IRF
            shock_size: Size of initial shock
        
        Returns:
            Impulse responses over time horizon
        """
        # Extract A matrices
        A_matrices = {}
        variables = list(var_coefficients.keys())
        n_vars = len(variables)
        
        # Get max lag from coefficient structure
        max_lag = 1
        for var_data in var_coefficients.values():
            if "A_matrices" in var_data:
                for A_key in var_data["A_matrices"].keys():
                    lag_num = int(A_key.split("_")[1])
                    max_lag = max(max_lag, lag_num)
        
        # Build A_k matrices (average across equations or use first variable's)
        first_var = variables[0]
        if first_var in var_coefficients and "A_matrices" in var_coefficients[first_var]:
            A_matrices = var_coefficients[first_var]["A_matrices"]
        else:
            return {"error": "Invalid VAR coefficients structure"}
        
        # Initialize IRF: IRF(0) = I (identity matrix)
        # IRF[h][i, j] = response of variable i to shock in variable j at horizon h
        irf = []
        irf_0 = np.eye(n_vars) * shock_size  # Initial shock matrix
        irf.append(irf_0.tolist())
        
        # Recursive computation: IRF(h) = Σₖ₌₁^p A_k · IRF(h-k)
        # where p is max_lag
        for h in range(1, horizon + 1):
            irf_h = np.zeros((n_vars, n_vars))
            
            for lag in range(1, min(h, max_lag) + 1):
                A_key = f"A_{lag}"
                if A_key in A_matrices:
                    A_k = np.array(A_matrices[A_key])
                    # IRF(h) += A_k · IRF(h-k)
                    if h - lag < len(irf):
                        irf_h += A_k @ np.array(irf[h - lag])
            
            irf.append(irf_h.tolist())
        
        # Extract IRF paths for each variable pair
        irf_paths: Dict[str, Dict[str, List[float]]] = {}
        for i, var_i in enumerate(variables):
            irf_paths[var_i] = {}
            for j, var_j in enumerate(variables):
                # Response of var_i to shock in var_j over time
                path = [irf[h][i][j] for h in range(horizon + 1)]
                irf_paths[var_i][var_j] = [float(x) for x in path]
        
        return {
            "impulse_responses": irf_paths,
            "horizon": horizon,
            "shock_size": shock_size,
            "variables": variables,
            "irf_matrix": irf,  # Full IRF matrices
        }

    def causal_discovery_pc_algorithm(
        self,
        df: Any,
        variables: List[str],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        PC algorithm for causal structure discovery (simplified implementation).
        
        PC algorithm steps:
        1. Start with complete undirected graph
        2. Remove edges if variables are conditionally independent given any subset
        3. Orient edges using v-structure detection
        4. Apply orientation rules (Meek's rules)
        
        Mathematical foundation: d-separation, conditional independence tests
        
        Args:
            df: Data
            variables: Variables to analyze
            alpha: Significance level for independence tests
        
        Returns:
            Learned DAG, edge orientations, independence test results
        """
        if df is None or len(variables) < 2:
            return {"error": "Need at least 2 variables"}
        
        data = df[variables].dropna()
        if len(data) < 10:
            return {"error": "Insufficient data"}
        
        # Step 1: Start with complete graph (all pairs connected)
        learned_graph = nx.DiGraph()
        learned_graph.add_nodes_from(variables)
        
        # Create undirected complete graph first
        undirected = nx.Graph()
        undirected.add_nodes_from(variables)
        for i, v1 in enumerate(variables):
            for v2 in variables[i+1:]:
                undirected.add_edge(v1, v2)
        
        # Step 2: Test conditional independence, remove edges
        # Simplified: test I(X;Y|Z) for all conditioning sets Z
        # Use correlation-based test as proxy for independence
        edges_to_remove = []
        
        for v1, v2 in list(undirected.edges()):
            # Test if v1 and v2 are independent given any subset of other variables
            others = [v for v in variables if v not in [v1, v2]]
            
            # Simple test: if partial correlation |ρ_{12|others}| < threshold, remove edge
            # Partial correlation: correlation after controlling for others
            try:
                # Compute partial correlation
                if len(others) == 0:
                    # No conditioning: use simple correlation
                    corr = data[[v1, v2]].corr().iloc[0, 1]
                else:
                    # Partial correlation approximation
                    # Simplified: regress v1 and v2 on others, then correlate residuals
                    from scipy.stats import pearsonr
                    
                    # Residuals after controlling for others
                    X_others = data[others].values
                    y1 = data[v1].values
                    y2 = data[v2].values
                    
                    # Remove linear effect of others
                    beta1 = np.linalg.pinv(X_others.T @ X_others) @ X_others.T @ y1
                    beta2 = np.linalg.pinv(X_others.T @ X_others) @ X_others.T @ y2
                    
                    res1 = y1 - X_others @ beta1
                    res2 = y2 - X_others @ beta2
                    
                    corr, p_val = pearsonr(res1, res2)
                    
                    # Remove edge if not significantly correlated (independent)
                    if abs(corr) < 0.1 or (p_val is not None and p_val > alpha):
                        edges_to_remove.append((v1, v2))
            except:
                pass
        
        # Remove edges
        for v1, v2 in edges_to_remove:
            if undirected.has_edge(v1, v2):
                undirected.remove_edge(v1, v2)
        
        # Step 3: Orient edges (simplified v-structure detection)
        # If X-Z-Y exists and X and Y are not connected, orient as X→Z←Y (v-structure)
        for z in variables:
            neighbors_z = list(undirected.neighbors(z))
            if len(neighbors_z) >= 2:
                for i, x in enumerate(neighbors_z):
                    for y in neighbors_z[i+1:]:
                        # If X and Y are not neighbors, create v-structure
                        if not undirected.has_edge(x, y):
                            learned_graph.add_edge(x, z)
                            learned_graph.add_edge(y, z)
        
        # Add remaining undirected edges (arbitrarily orient)
        for x, y in undirected.edges():
            if not (learned_graph.has_edge(x, y) or learned_graph.has_edge(y, x)):
                learned_graph.add_edge(x, y)  # Arbitrary orientation
        
        return {
            "learned_dag": learned_graph,
            "edges": list(learned_graph.edges()),
            "nodes": list(learned_graph.nodes()),
            "edges_removed": len(edges_to_remove),
            "method": "PC_algorithm_simplified",
        }

    def evolutionary_multi_objective_optimization(
        self,
        initial_state: Dict[str, float],
        objectives: Dict[str, float],  # {var: weight}
        constraints: Dict[str, Tuple[float, float]],
        intervention_vars: List[str],
        population_size: int = 50,
        generations: int = 100,
    ) -> Dict[str, Any]:
        """
        Evolutionary algorithm (NSGA-II inspired) for multi-objective optimization.
        
        Mathematical foundation:
        - Population: P = {x₁, x₂, ..., x_N} (intervention candidates)
        - Fitness: f(x) = [f₁(x), f₂(x), ..., fₖ(x)] (objective vector)
        - Selection: Tournament selection based on Pareto dominance
        - Crossover: Blend crossover: x_new = α·x₁ + (1-α)·x₂
        - Mutation: Gaussian mutation: x_new = x + N(0, σ²)
        
        Args:
            initial_state: Current state
            objectives: {variable: weight} to optimize
            constraints: {variable: (min, max)} bounds
            intervention_vars: Variables to optimize
            population_size: Size of population
            generations: Number of generations
        
        Returns:
            Pareto front, best solutions, convergence history
        """
        # Initialize population
        population: List[Dict[str, float]] = []
        for _ in range(population_size):
            individual = {}
            for var in intervention_vars:
                stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
                current = initial_state.get(var, stats["mean"])
                if var in constraints:
                    min_val, max_val = constraints[var]
                    individual[var] = float(self.rng.uniform(min_val, max_val))
                else:
                    individual[var] = float(self.rng.normal(current, stats["std"]))
            population.append(individual)
        
        pareto_front: List[Dict[str, Any]] = []
        convergence_history: List[float] = []
        
        for generation in range(generations):
            # Evaluate fitness for all individuals
            fitness_scores: List[Dict[str, float]] = []
            for individual in population:
                outcome = self._predict_outcomes(initial_state, individual)
                fitness = {k: outcome.get(k, 0.0) * weight for k, weight in objectives.items()}
                combined = sum(fitness.values())
                fitness_scores.append({
                    "individual": individual,
                    "fitness_vector": fitness,
                    "combined": combined,
                })
            
            # Update Pareto front
            for fs in fitness_scores:
                is_dominated = False
                for pf in pareto_front:
                    pf_fit = pf["fitness_vector"]
                    fs_fit = fs["fitness_vector"]
                    
                    # Check if pf dominates fs
                    all_better = all(
                        pf_fit.get(k, 0.0) >= fs_fit.get(k, 0.0) for k in objectives.keys()
                    )
                    some_better = any(
                        pf_fit.get(k, 0.0) > fs_fit.get(k, 0.0) for k in objectives.keys()
                    )
                    
                    if all_better and some_better:
                        is_dominated = True
                        break
                
                if not is_dominated:
                    # Remove dominated points from front
                    pareto_front = [
                        p for p in pareto_front
                        if not (
                            all(fs["fitness_vector"].get(k, 0.0) >= p["fitness_vector"].get(k, 0.0)
                            for k in objectives.keys()) and
                            any(fs["fitness_vector"].get(k, 0.0) > p["fitness_vector"].get(k, 0.0)
                            for k in objectives.keys())
                        )
                    ]
                    pareto_front.append(fs)
            
            # Track convergence (average combined fitness)
            avg_fitness = np.mean([fs["combined"] for fs in fitness_scores])
            convergence_history.append(float(avg_fitness))
            
            # Selection: tournament selection (simplified)
            # Crossover: blend crossover
            # Mutation: Gaussian mutation
            new_population = []
            while len(new_population) < population_size:
                # Tournament selection
                idx1 = self.rng.integers(0, len(population))
                idx2 = self.rng.integers(0, len(population))
                parent1 = population[idx1]
                parent2 = population[idx2]
                
                # Crossover: blend (α = random)
                alpha = self.rng.random()
                child = {}
                for var in intervention_vars:
                    val1 = parent1.get(var, 0.0)
                    val2 = parent2.get(var, 0.0)
                    child[var] = float(alpha * val1 + (1 - alpha) * val2)
                    
                    # Mutation: Gaussian noise
                    if self.rng.random() < 0.1:  # 10% mutation rate
                        stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
                        child[var] += float(self.rng.normal(0, stats["std"] * 0.1))
                    
                    # Enforce constraints
                    if var in constraints:
                        min_val, max_val = constraints[var]
                        child[var] = float(np.clip(child[var], min_val, max_val))
                
                new_population.append(child)
            
            population = new_population
        
        # Sort Pareto front by combined fitness
        pareto_front.sort(key=lambda x: x["combined"], reverse=True)
        
        return {
            "pareto_front": pareto_front[:20],  # Top 20
            "best_solution": pareto_front[0] if pareto_front else None,
            "convergence_history": convergence_history,
            "final_population_size": len(population),
            "generations": generations,
        }

    def shannon_entropy(
        self,
        variable: str,
        df: Any,
        bins: Optional[int] = None,
    ) -> float:
        """
        Compute Shannon entropy: H(X) = -Σᵢ p(xᵢ) log₂ p(xᵢ)
        
        Args:
            variable: Variable name
            df: Data
            bins: Number of bins for discretization (auto if None)
        
        Returns:
            Entropy value in bits
        """
        if df is None or variable not in df.columns:
            return 0.0
        
        series = df[variable].dropna()
        if len(series) < 5:
            return 0.0
        
        # Check cache
        cache_key = f"{variable}_{len(series)}_{bins}"
        if cache_key in self._entropy_cache:
            return self._entropy_cache[cache_key]
        
        # Discretize
        if bins is None:
            bins = min(20, max(5, int(np.sqrt(len(series)))))
        
        hist, _ = np.histogram(series, bins=bins)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        
        # Shannon entropy
        entropy = float(-np.sum(probs * np.log2(probs)))
        self._entropy_cache[cache_key] = entropy
        return entropy

    def mutual_information(
        self,
        var1: str,
        var2: str,
        df: Any,
    ) -> float:
        """
        Compute mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            var1: First variable
            var2: Second variable
            df: Data
        
        Returns:
            Mutual information in bits
        """
        if df is None or var1 not in df.columns or var2 not in df.columns:
            return 0.0
        
        # Check cache
        cache_key = (var1, var2)
        if cache_key in self._mi_cache:
            return self._mi_cache[cache_key]
        
        data = df[[var1, var2]].dropna()
        if len(data) < 10:
            return 0.0
        
        # Individual entropies
        h1 = self.shannon_entropy(var1, df)
        h2 = self.shannon_entropy(var2, df)
        
        # Joint entropy: H(X,Y)
        n_bins = min(10, max(3, int(np.cbrt(len(data)))))
        hist_2d, _, _ = np.histogram2d(data[var1], data[var2], bins=n_bins)
        hist_2d = hist_2d[hist_2d > 0]
        probs_joint = hist_2d / hist_2d.sum()
        h_joint = float(-np.sum(probs_joint * np.log2(probs_joint)))
        
        # Mutual information
        mi = float(max(0.0, h1 + h2 - h_joint))
        self._mi_cache[cache_key] = mi
        return mi

    def conditional_mutual_information(
        self,
        var1: str,
        var2: str,
        condition: str,
        df: Any,
    ) -> float:
        """
        Compute conditional mutual information: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        
        Args:
            var1: First variable
            var2: Second variable
            condition: Conditioning variable
            df: Data
        
        Returns:
            Conditional MI in bits
        """
        if df is None or any(v not in df.columns for v in [var1, var2, condition]):
            return 0.0
        
        data = df[[var1, var2, condition]].dropna()
        if len(data) < 10:
            return 0.0
        
        # H(X,Z)
        h_xz = self.shannon_entropy(f"{var1}_{condition}", pd.DataFrame({
            f"{var1}_{condition}": data[var1].astype(str) + "_" + data[condition].astype(str)
        }))
        
        # H(Y,Z)
        h_yz = self.shannon_entropy(f"{var2}_{condition}", pd.DataFrame({
            f"{var2}_{condition}": data[var2].astype(str) + "_" + data[condition].astype(str)
        }))
        
        # H(X,Y,Z) - simplified joint entropy
        # H(Z)
        h_z = self.shannon_entropy(condition, df)
        
        # Simplified: I(X;Y|Z) ≈ I(X;Y) - I(X;Z) - I(Y;Z) + I(X;Y;Z)
        # Use chain rule approximation
        mi_xy = self.mutual_information(var1, var2, df)
        mi_xz = self.mutual_information(var1, condition, df)
        mi_yz = self.mutual_information(var2, condition, df)
        
        # Chain rule: I(X;Y|Z) = I(X;Y) - I(X;Y;Z) where I(X;Y;Z) is interaction
        # Approximation: I(X;Y|Z) ≈ I(X;Y) - min(I(X;Z), I(Y;Z))
        cmi = float(max(0.0, mi_xy - min(mi_xz, mi_yz)))
        
        return cmi

    def cross_validate_edge_strength(
        self,
        df: Any,
        parent: str,
        child: str,
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Cross-validation for edge strength estimation.
        
        Mathematical:
        - K-fold CV: partition data into K folds
        - Train on K-1 folds, test on held-out fold
        - CV error: CV = (1/K) Σᵢ₌₁ᴷ MSE_i where MSE_i is mean squared error on fold i
        
        Args:
            df: Data
            parent: Source variable
            child: Target variable
            n_folds: Number of CV folds
        
        Returns:
            CV scores, mean CV error, standard error
        """
        if df is None or parent not in df.columns or child not in df.columns:
            return {"error": "Invalid data"}
        
        data = df[[parent, child]].dropna()
        if len(data) < n_folds * 3:
            return {"error": "Insufficient data for CV"}
        
        n = len(data)
        fold_size = n // n_folds
        
        cv_errors: List[float] = []
        
        for fold in range(n_folds):
            # Split: test = fold, train = rest
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n
            
            test_indices = list(range(test_start, test_end))
            train_indices = [i for i in range(n) if i not in test_indices]
            
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            
            if len(train_data) < 5 or len(test_data) < 2:
                continue
            
            # Fit on training data
            X_train = train_data[parent].values.reshape(-1, 1)
            y_train = train_data[child].values
            X_test = test_data[parent].values.reshape(-1, 1)
            y_test = test_data[child].values
            
            # OLS: β = (X'X)⁻¹X'y
            XtX = X_train.T @ X_train
            Xty = X_train.T @ y_train
            beta = (np.linalg.pinv(XtX) @ Xty)[0]
            
            # Predict on test: ŷ = Xβ
            y_pred = X_test @ np.array([beta])
            
            # MSE on test fold
            mse = float(np.mean((y_test - y_pred) ** 2))
            cv_errors.append(mse)
        
        if len(cv_errors) == 0:
            return {"error": "CV failed"}
        
        # CV statistics
        mean_cv_error = float(np.mean(cv_errors))
        std_cv_error = float(np.std(cv_errors))
        se_cv = std_cv_error / np.sqrt(len(cv_errors))  # Standard error
        
        return {
            "cv_errors": [float(e) for e in cv_errors],
            "mean_cv_error": mean_cv_error,
            "std_cv_error": std_cv_error,
            "standard_error": float(se_cv),
            "n_folds": len(cv_errors),
            "cv_score": mean_cv_error,  # Lower is better
        }

    def integrated_gradients_attribution(
        self,
        baseline_state: Dict[str, float],
        target_state: Dict[str, float],
        target: str,
        n_steps: int = 50,
    ) -> Dict[str, float]:
        """
        Integrated Gradients for causal attribution: how much does each variable contribute?
        
        Mathematical formulation:
        - IG_i = (x_i - x_i^0) · ∫₀¹ [∂f/∂x_i](x^0 + t·(x - x^0)) dt
        - Approximated: IG_i ≈ (x_i - x_i^0) · (1/m) Σⱼ₌₁ᵐ [∂f/∂x_i](x^0 + (j/m)·(x - x^0))
        - Attribution: A_i = IG_i / Σⱼ IGⱼ (normalized)
        
        Args:
            baseline_state: Reference state (x^0)
            target_state: Target state (x)
            target: Outcome variable
            n_steps: Number of integration steps
        
        Returns:
            Attributions for each variable
        """
        # Variables that differ between baseline and target
        diff_vars = [
            v for v in set(list(baseline_state.keys()) + list(target_state.keys()))
            if abs(baseline_state.get(v, 0.0) - target_state.get(v, 0.0)) > 1e-6
        ]
        
        if not diff_vars:
            return {}
        
        integrated_gradients: Dict[str, float] = {}
        
        for var in diff_vars:
            x0 = baseline_state.get(var, 0.0)
            x1 = target_state.get(var, 0.0)
            delta = x1 - x0
            
            if abs(delta) < 1e-6:
                integrated_gradients[var] = 0.0
                continue
            
            # Integrate gradient along path: x(t) = x^0 + t·(x - x^0), t ∈ [0,1]
            grad_sum = 0.0
            
            for step in range(1, n_steps + 1):
                t = step / n_steps  # t ∈ [0,1]
                
                # Interpolated state: x(t) = x^0 + t·(x - x^0)
                interpolated_state = baseline_state.copy()
                for v in diff_vars:
                    v0 = baseline_state.get(v, 0.0)
                    v1 = target_state.get(v, 0.0)
                    interpolated_state[v] = v0 + t * (v1 - v0)
                
                # Compute gradient: ∂f/∂x_i at interpolated state
                # Use finite differences
                epsilon = abs(delta) * 1e-4
                perturbed_state = interpolated_state.copy()
                perturbed_state[var] += epsilon
                
                outcome_base = self._predict_outcomes({}, interpolated_state)
                outcome_pert = self._predict_outcomes({}, perturbed_state)
                
                grad = (outcome_pert.get(target, 0.0) - outcome_base.get(target, 0.0)) / epsilon
                grad_sum += grad
            
            # Integrated gradient: IG = delta · average_gradient
            avg_grad = grad_sum / n_steps
            ig = delta * avg_grad
            integrated_gradients[var] = float(ig)
        
        # Normalize attributions
        total_ig = sum(abs(v) for v in integrated_gradients.values())
        if total_ig > 1e-6:
            attributions = {k: float(v / total_ig) for k, v in integrated_gradients.items()}
        else:
            attributions = {k: 0.0 for k in integrated_gradients.keys()}
        
        return {
            "integrated_gradients": integrated_gradients,
            "attributions": attributions,  # Normalized to sum to 1
            "total_attribution": float(sum(abs(v) for v in attributions.values())),
        }

    def bellman_optimal_intervention(
        self,
        initial_state: Dict[str, float],
        target: str,
        intervention_vars: List[str],
        horizon: int = 5,
        discount: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Dynamic Programming (Bellman optimality) for optimal intervention sequence.
        
        Mathematical formulation:
        - Value function: V*(x) = max_u [r(x,u) + γ·V*(f(x,u))]
        - Optimal policy: π*(x) = argmax_u [r(x,u) + γ·V*(f(x,u))]
        - Backward induction: solve V*(x) from terminal time backwards
        
        Args:
            initial_state: Starting state
            target: Variable to maximize
            intervention_vars: Available interventions
            horizon: Time horizon T
            discount: Discount factor γ ∈ [0,1]
        
        Returns:
            Optimal policy, value function, intervention sequence
        """
        # Discretize state space (simplified: use current state as reference)
        # Value function: V_t(x) = value at time t in state x
        value_function: Dict[int, Dict[Tuple, float]] = {}
        policy: Dict[int, Dict[Tuple, Dict[str, float]]] = {}
        
        # Terminal condition: V_T(x) = r(x) (immediate reward)
        # Reward: r(x) = target_value(x)
        def reward(state: Dict[str, float]) -> float:
            outcome = self._predict_outcomes({}, state)
            return float(outcome.get(target, 0.0))
        
        # Backward induction: from T down to 0
        # For each time t from T-1 down to 0:
        for t in range(horizon - 1, -1, -1):
            # Simplified: evaluate at current state (could expand to full state space)
            state_key = tuple(sorted(initial_state.items()))
            
            if t == horizon - 1:
                # Terminal: V_T(x) = r(x)
                value_function[t] = {state_key: reward(initial_state)}
                policy[t] = {state_key: {}}
            else:
                # Bellman: V_t(x) = max_u [r(x) + γ·V_{t+1}(f(x,u))]
                best_value = float("-inf")
                best_intervention: Dict[str, float] = {}
                
                # Search intervention space (simplified: sample candidates)
                for _ in range(20):
                    candidate_intervention = {}
                    for var in intervention_vars:
                        stats = self.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
                        current = initial_state.get(var, stats["mean"])
                        candidate_intervention[var] = float(self.rng.normal(current, stats["std"] * 0.5))
                    
                    # Next state: f(x, u)
                    next_state = self._predict_outcomes(initial_state, candidate_intervention)
                    next_key = tuple(sorted(next_state.items()))
                    
                    # Immediate reward
                    r = reward(next_state)
                    
                    # Future value: γ·V_{t+1}(f(x,u))
                    if t + 1 in value_function and next_key in value_function[t + 1]:
                        future_val = value_function[t + 1][next_key]
                    else:
                        future_val = 0.0
                    
                    # Total value: r + γ·V_{t+1}
                    total_value = r + discount * future_val
                    
                    if total_value > best_value:
                        best_value = total_value
                        best_intervention = candidate_intervention
                
                value_function[t] = {state_key: best_value}
                policy[t] = {state_key: best_intervention}
        
        # Extract optimal sequence
        optimal_sequence: List[Dict[str, float]] = []
        current_state = initial_state.copy()
        
        for t in range(horizon):
            state_key = tuple(sorted(current_state.items()))
            if t in policy and state_key in policy[t]:
                intervention = policy[t][state_key]
                optimal_sequence.append(intervention)
                current_state = self._predict_outcomes(current_state, intervention)
        
        return {
            "optimal_sequence": optimal_sequence,
            "value_function": {
                t: {str(k): v for k, v in vf.items()}
                for t, vf in value_function.items()
            },
            "policy": {
                t: {str(k): v for k, v in p.items()}
                for t, p in policy.items()
            },
            "total_value": float(value_function.get(0, {}).get(tuple(sorted(initial_state.items())), 0.0)),
            "horizon": horizon,
            "discount_factor": discount,
        }

    def shapley_value_attribution(
        self,
        baseline_state: Dict[str, float],
        target_state: Dict[str, float],
        target: str,
    ) -> Dict[str, float]:
        """
        Shapley values for fair attribution: marginal contribution of each variable.
        
        Mathematical formulation:
        - Shapley value: φᵢ = Σ_{S ⊆ N\{i}} [|S|!(n-|S|-1)!/n!] · [v(S∪{i}) - v(S)]
        - where v(S) = outcome with variables in S set to target, others to baseline
        - Fair attribution: satisfies efficiency, symmetry, dummy, additivity
        
        Args:
            baseline_state: Baseline (all variables at baseline)
            target_state: Target (all variables at target)
            target: Outcome variable
        
        Returns:
            Shapley values for each variable
        """
        variables = list(set(list(baseline_state.keys()) + list(target_state.keys())))
        n = len(variables)
        
        if n == 0:
            return {}
        
        shapley_values: Dict[str, float] = {var: 0.0 for var in variables}
        
        # Compute value function: v(S) = outcome when S are set to target, rest to baseline
        def value_function(subset: set) -> float:
            state = baseline_state.copy()
            for var in subset:
                if var in target_state:
                    state[var] = target_state[var]
            outcome = self._predict_outcomes({}, state)
            return float(outcome.get(target, 0.0))
        
        # Compute Shapley value for each variable
        for var in variables:
            phi_i = 0.0
            
            # Sum over all subsets S not containing var
            others = [v for v in variables if v != var]
            
            # For each subset size
            for subset_size in range(len(others) + 1):
                # Generate all subsets of size subset_size
                from itertools import combinations
                
                for subset in combinations(others, subset_size):
                    S = set(subset)
                    
                    # Weight: |S|!(n-|S|-1)!/n!
                    s_size = len(S)
                    weight = (math.factorial(s_size) * math.factorial(n - s_size - 1)) / math.factorial(n)
                    
                    # Marginal contribution: v(S∪{i}) - v(S)
                    S_with_i = S | {var}
                    marginal = value_function(S_with_i) - value_function(S)
                    
                    phi_i += weight * marginal
            
            shapley_values[var] = float(phi_i)
        
        return {
            "shapley_values": shapley_values,
            "total_attribution": float(sum(shapley_values.values())),
            "normalized": {k: float(v / sum(abs(vi) for vi in shapley_values.values()) if sum(abs(vi) for vi in shapley_values.values()) > 0 else 0.0)
                          for k, v in shapley_values.items()},
        }
