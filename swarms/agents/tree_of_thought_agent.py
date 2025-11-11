"""
Tree-of-Thought (ToT) Reasoning Framework Implementation.

This module implements a comprehensive Tree-of-Thought reasoning system based on
the formal framework where we introduce a tree-structured latent variable R
representing multiple candidate reasoning paths.

Mathematical Foundation:
    Core Probabilistic Model:
        p_θ(y | x) = Σ_{R ∈ T} p_θ(R | x) · p_θ(y | R, x)
    
    Where:
    - x = input (question, task description) ∈ X
    - y = final answer ∈ Y
    - R = {r^(1), r^(2), ..., r^(k)} = set of candidate reasoning paths
    - T = set of reasoning trees
    - θ = model parameters
    
    Tree Structure:
        T = (V, E) where:
        - V = {v₁, v₂, ..., v_n} = nodes (thoughts)
        - E = {(v_i, v_j) | v_i → v_j} = edges (reasoning transitions)
        - Root: v_root = initial problem state
        - Leaves: L = {v | children(v) = ∅}
    
    Path Probability:
        P(path = (v₀, v₁, ..., v_k)) = Π_{i=0}^{k-1} P(v_{i+1} | v_i, x)
        
        Where P(v_{i+1} | v_i, x) is the transition probability.
    
    Marginalization over Tree:
        p_θ(y | x) = Σ_{path ∈ paths(T)} P(path) · p_θ(y | path, x)
        
        Where paths(T) is the set of all root-to-leaf paths.
    
    Information-Theoretic Tree Search:
        Information gain at node v:
            I(v; Y | x) = H(Y | x) - H(Y | v, x)
        
        Expected information gain:
            E[I(v; Y | x)] = Σ_{child} P(child | v) · I(child; Y | x)
    
    Quantum Tree Superposition:
        |ψ_tree⟩ = Σ_{path} α_path |path⟩ ⊗ |y_path⟩
        
        Where:
        - α_path = √(P(path)) = amplitude for path
        - |path⟩ = quantum state representing reasoning path
        - |y_path⟩ = answer state for path
        
        Measurement probability:
            P(y | x) = |⟨y | ψ_tree⟩|² = |Σ_{path: y_path=y} α_path|²
    
    Monte Carlo Tree Search (MCTS):
        UCB1 formula:
            UCB1(v) = Q(v) + c · √(ln(N(v_parent)) / N(v))
        
        Where:
        - Q(v) = average value: Q(v) = (1/N(v)) Σ_{i=1}^{N(v)} V_i
        - N(v) = visit count
        - c = exploration constant (typically √2)
        - V_i = evaluation value from simulation i
        
        Value backpropagation:
            Q(v) ← (N(v) · Q(v) + V_new) / (N(v) + 1)
            N(v) ← N(v) + 1
        
        Selection policy:
            v* = argmax_{v ∈ children(v_parent)} UCB1(v)
    
    Beam Search (Pruned Tree Search):
        Beam width B, keep top-B nodes at each depth:
            Beam_d = {v | v ∈ Top_B(score(v), v ∈ candidates_d)}
        
        Score function:
            score(v) = α · heuristic(v) + β · depth_penalty(v) + γ · path_prob(v)
        
        Where:
        - heuristic(v) = evaluator score
        - depth_penalty(v) = -λ · depth(v)
        - path_prob(v) = log P(path_to_v)
    
    Statistical Mechanics (Tree Energy):
        Energy of path:
            E(path, x) = -log P(path | x) = -Σ_{i} log P(v_{i+1} | v_i, x)
        
        Boltzmann distribution over paths:
            P(path | x) = (1/Z(x)) exp(-E(path, x) / T)
        
        Partition function:
            Z(x) = Σ_{path ∈ paths(T)} exp(-E(path, x) / T)
        
        Free energy:
            F(x) = -T log Z(x)
    
    Graph-Theoretic Properties:
        Tree depth: D = max_{path} |path|
        Branching factor: b = avg_{v} |children(v)|
        Tree size: |T| = Σ_{d=0}^D b^d (for balanced tree)
        
        Path diversity:
            Diversity(T) = (1/|L|) Σ_{l₁, l₂ ∈ L} distance(l₁, l₂)
        
        Where distance is edit distance or semantic distance.
    
    Optimization Objective:
        Best path selection:
            path* = argmax_{path} [log p_θ(y | path, x) + λ · log P(path | x)]
        
        Multi-objective:
            path* = argmax_{path} [w₁ · correctness + w₂ · efficiency + w₃ · diversity]
    
    Computational Complexity:
        Time: O(b^D · (expand_cost + eval_cost))
        - b = branching factor
        - D = max depth
        - expand_cost = cost to generate children
        - eval_cost = cost to evaluate node
        
        With beam search (width B):
            Time: O(B · D · (expand_cost + eval_cost))
        
        With MCTS (N simulations):
            Time: O(N · (selection_cost + expand_cost + eval_cost + backprop_cost))
    
    Variational Tree Inference:
        ELBO for tree search:
            log p_θ(y | x) ≥ E_{q_φ(path|x,y)}[log p_θ(y | path, x)] - KL(q_φ(path|x,y) || p_θ(path|x))
        
        Where q_φ is the search policy (beam search, MCTS, etc.)

At inference time:
    1. Build a tree of reasoning paths by expanding nodes
    2. Evaluate partial reasoning paths using heuristic
    3. Search the tree using beam search or MCTS
    4. Extract final answer from best leaf node
    
    Search strategies:
    - Beam search: Top-B paths at each depth
    - MCTS: UCB1-based exploration-exploitation
    - BFS/DFS: Exhaustive or depth-limited search
    - Quantum: Superposition of all paths with measurement
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from uuid import uuid4, UUID
import math
import random
from collections import deque, Counter

from loguru import logger


class SearchStrategy(str, Enum):
    """Search strategies for exploring the reasoning tree."""
    
    BEAM = "beam"
    BFS = "bfs"
    DFS = "dfs"
    MCTS = "mcts"
    QUANTUM = "quantum"  # Quantum-inspired tree search


class TreeInformationTheory:
    """
    Information-theoretic utilities for tree reasoning analysis.
    
    Implements entropy, information gain, and tree diversity measures.
    """
    
    @staticmethod
    def path_entropy(path_probs: List[float]) -> float:
        """
        Calculate entropy of path distribution: H(Path | X) = -Σ P(path) log P(path).
        
        Args:
            path_probs: List of path probabilities
            
        Returns:
            Entropy value in bits
        """
        if not path_probs:
            return 0.0
        
        # Normalize
        total = sum(path_probs)
        if total == 0:
            return 0.0
        
        normalized = [p / total for p in path_probs]
        
        # Calculate entropy
        h = 0.0
        for p in normalized:
            if p > 0:
                h -= p * math.log2(p)
        
        return h
    
    @staticmethod
    def information_gain(
        prior_entropy: float,
        conditional_entropy: float
    ) -> float:
        """
        Calculate information gain: I(V; Y | X) = H(Y | X) - H(Y | V, X).
        
        Args:
            prior_entropy: H(Y | X) - entropy before observing node
            conditional_entropy: H(Y | V, X) - entropy after observing node
            
        Returns:
            Information gain value
        """
        return prior_entropy - conditional_entropy
    
    @staticmethod
    def expected_information_gain(
        node_probs: List[float],
        child_entropies: List[float]
    ) -> float:
        """
        Calculate expected information gain: E[I(v; Y | x)] = Σ P(child) · I(child; Y | x).
        
        Args:
            node_probs: Probabilities of child nodes P(child | v)
            child_entropies: Information gains for each child
            
        Returns:
            Expected information gain
        """
        if not node_probs or not child_entropies:
            return 0.0
        
        if len(node_probs) != len(child_entropies):
            return 0.0
        
        # Normalize probabilities
        total = sum(node_probs)
        if total == 0:
            return 0.0
        
        normalized = [p / total for p in node_probs]
        
        # Weighted sum
        expected_gain = sum(p * gain for p, gain in zip(normalized, child_entropies))
        return expected_gain
    
    @staticmethod
    def tree_diversity(leaf_paths: List[List[str]]) -> float:
        """
        Calculate tree diversity: Diversity(T) = (1/|L|) Σ distance(l₁, l₂).
        
        Args:
            leaf_paths: List of paths to leaves (each path is list of node texts)
            
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(leaf_paths) < 2:
            return 0.0
        
        total_distance = 0.0
        pairs = 0
        
        for i in range(len(leaf_paths)):
            for j in range(i + 1, len(leaf_paths)):
                # Calculate edit distance (simplified: use text similarity)
                path1 = " ".join(leaf_paths[i])
                path2 = " ".join(leaf_paths[j])
                distance = TreeInformationTheory._edit_distance(path1, path2)
                total_distance += distance
                pairs += 1
        
        if pairs == 0:
            return 0.0
        
        return total_distance / pairs
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> float:
        """
        Calculate normalized edit distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Normalized edit distance [0, 1]
        """
        if not s1 and not s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
        
        # Simple character-level edit distance
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
        
        max_len = max(m, n)
        return dp[m][n] / max_len if max_len > 0 else 0.0


class QuantumTreeSearch:
    """
    Quantum-inspired tree search with superposition of paths.
    
    Implements: |ψ_tree⟩ = Σ_path α_path |path⟩
    """
    
    @staticmethod
    def calculate_path_amplitudes(path_probs: List[float]) -> List[float]:
        """
        Calculate quantum amplitudes: α_path = √(P(path)).
        
        Args:
            path_probs: List of path probabilities
            
        Returns:
            List of amplitudes
        """
        return [math.sqrt(max(0.0, p)) for p in path_probs]
    
    @staticmethod
    def quantum_measurement(
        paths: List[List[str]],
        answers: List[str],
        path_probs: Optional[List[float]] = None
    ) -> Tuple[str, float]:
        """
        Quantum measurement: P(y | x) = |⟨y | ψ_tree⟩|² = |Σ_{path: y_path=y} α_path|².
        
        Args:
            paths: List of reasoning paths (each path is list of node texts)
            answers: List of answers corresponding to paths
            path_probs: Optional path probabilities (uniform if None)
            
        Returns:
            Tuple of (most likely answer, probability)
        """
        if not paths or not answers:
            return "", 0.0
        
        if path_probs is None:
            path_probs = [1.0 / len(paths)] * len(paths)
        
        # Calculate amplitudes
        amplitudes = QuantumTreeSearch.calculate_path_amplitudes(path_probs)
        
        # Group by answer and sum amplitudes
        answer_amplitudes: Dict[str, float] = {}
        for answer, amp in zip(answers, amplitudes):
            normalized_answer = answer.lower().strip()
            answer_amplitudes[normalized_answer] = answer_amplitudes.get(normalized_answer, 0.0) + amp
        
        # Calculate probabilities: |amplitude|²
        answer_probs = {ans: amp ** 2 for ans, amp in answer_amplitudes.items()}
        
        # Normalize
        total = sum(answer_probs.values())
        if total > 0:
            answer_probs = {ans: prob / total for ans, prob in answer_probs.items()}
        
        # Return most likely answer
        if answer_probs:
            best_answer = max(answer_probs.items(), key=lambda x: x[1])
            return best_answer[0], best_answer[1]
        
        return "", 0.0
    
    @staticmethod
    def quantum_tree_sampling(
        nodes: List[Any],
        path_probs: List[float],
        num_samples: int = 1
    ) -> List[Any]:
        """
        Sample nodes using quantum-inspired superposition.
        
        Args:
            nodes: List of nodes to sample from
            path_probs: Probabilities for each node's path
            num_samples: Number of samples to generate
            
        Returns:
            List of sampled nodes
        """
        if not nodes:
            return []
        
        # Normalize probabilities
        total = sum(path_probs)
        if total == 0:
            path_probs = [1.0 / len(nodes)] * len(nodes)
        else:
            path_probs = [p / total for p in path_probs]
        
        # Calculate amplitudes
        amplitudes = QuantumTreeSearch.calculate_path_amplitudes(path_probs)
        
        # Sample based on amplitude squared
        probs = [amp ** 2 for amp in amplitudes]
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        
        # Sample
        sampled_indices = random.choices(
            range(len(nodes)),
            weights=probs,
            k=num_samples
        )
        
        return [nodes[i] for i in sampled_indices]


class TreeEnergyFunction:
    """
    Energy-based functions for tree reasoning (statistical mechanics).
    
    Implements: E(path, x) = -log P(path | x)
    """
    
    @staticmethod
    def calculate_path_energy(path_logprob: float) -> float:
        """
        Calculate energy of path: E(path, x) = -log P(path | x).
        
        Args:
            path_logprob: Log probability of path
            
        Returns:
            Energy value
        """
        return -path_logprob
    
    @staticmethod
    def boltzmann_path_weight(energy: float, temperature: float) -> float:
        """
        Calculate Boltzmann weight: w(path) = exp(-E(path, x) / T).
        
        Args:
            energy: Energy value E(path, x)
            temperature: Temperature parameter T
            
        Returns:
            Boltzmann weight
        """
        if temperature <= 0:
            return 0.0 if energy > 0 else 1.0
        
        return math.exp(-energy / temperature)
    
    @staticmethod
    def tree_partition_function(
        path_energies: List[float],
        temperature: float
    ) -> float:
        """
        Calculate partition function: Z(x) = Σ_{path} exp(-E(path, x) / T).
        
        Args:
            path_energies: List of energy values for paths
            temperature: Temperature parameter T
            
        Returns:
            Partition function value
        """
        if temperature <= 0:
            return 1.0
        
        weights = [TreeEnergyFunction.boltzmann_path_weight(e, temperature) for e in path_energies]
        return sum(weights)
    
    @staticmethod
    def tree_free_energy(partition_function: float, temperature: float) -> float:
        """
        Calculate free energy: F(x) = -T log Z(x).
        
        Args:
            partition_function: Partition function Z(x)
            temperature: Temperature parameter T
            
        Returns:
            Free energy value
        """
        if partition_function <= 0:
            return float('inf')
        
        if temperature <= 0:
            return 0.0
        
        return -temperature * math.log(partition_function)
    
    @staticmethod
    def boltzmann_path_sampling(
        paths: List[Any],
        path_logprobs: List[float],
        temperature: float,
        num_samples: int = 1
    ) -> List[Any]:
        """
        Sample paths using Boltzmann distribution.
        
        Args:
            paths: List of paths to sample from
            path_logprobs: Log probabilities for each path
            temperature: Temperature parameter T
            num_samples: Number of samples to generate
            
        Returns:
            List of sampled paths
        """
        if not paths:
            return []
        
        # Calculate energies
        energies = [TreeEnergyFunction.calculate_path_energy(logprob) for logprob in path_logprobs]
        
        # Calculate partition function
        z = TreeEnergyFunction.tree_partition_function(energies, temperature)
        
        if z <= 0:
            return random.sample(paths, min(num_samples, len(paths)))
        
        # Calculate Boltzmann weights
        weights = [
            TreeEnergyFunction.boltzmann_path_weight(e, temperature) / z
            for e in energies
        ]
        
        # Sample
        sampled_indices = random.choices(
            range(len(paths)),
            weights=weights,
            k=num_samples
        )
        
        return [paths[i] for i in sampled_indices]


class TreeGraphTheory:
    """
    Graph-theoretic utilities for tree reasoning.
    
    Implements tree properties, path finding, and optimization.
    """
    
    @staticmethod
    def calculate_path_probability(
        node_scores: List[float],
        normalize: bool = True
    ) -> float:
        """
        Calculate path probability: P(path) = Π P(v_{i+1} | v_i, x).
        
        Args:
            node_scores: Scores/probabilities for each node in path
            normalize: Whether to normalize scores to probabilities
            
        Returns:
            Path probability
        """
        if not node_scores:
            return 0.0
        
        if normalize:
            # Convert scores to probabilities using softmax
            max_score = max(node_scores)
            exp_scores = [math.exp(s - max_score) for s in node_scores]
            total = sum(exp_scores)
            if total > 0:
                probs = [s / total for s in exp_scores]
            else:
                probs = [1.0 / len(node_scores)] * len(node_scores)
        else:
            probs = node_scores
        
        # Product of probabilities
        path_prob = 1.0
        for prob in probs:
            path_prob *= max(0.0, min(1.0, prob))
        
        return path_prob
    
    @staticmethod
    def find_optimal_path(
        paths: List[List[Any]],
        path_scores: List[float],
        path_lengths: List[int],
        lambda_reg: float = 0.1
    ) -> Optional[List[Any]]:
        """
        Find optimal path: path* = argmax [score(path) - λ · length(path)].
        
        Args:
            paths: List of paths (each path is list of nodes)
            path_scores: Scores for each path
            path_lengths: Lengths of each path
            lambda_reg: Regularization parameter λ
            
        Returns:
            Optimal path, or None if empty
        """
        if not paths:
            return None
        
        best_path = None
        best_cost = float('-inf')
        
        for path, score, length in zip(paths, path_scores, path_lengths):
            # Cost = score - λ * length (higher is better)
            cost = score - lambda_reg * length
            
            if cost > best_cost:
                best_cost = cost
                best_path = path
        
        return best_path
    
    @staticmethod
    def calculate_tree_metrics(
        nodes: List[Any],
        get_depth: Callable[[Any], int],
        get_children: Callable[[Any], List[Any]]
    ) -> Dict[str, float]:
        """
        Calculate tree metrics: depth, branching factor, size.
        
        Args:
            nodes: List of all nodes in tree
            get_depth: Function to get depth of node
            get_children: Function to get children of node
            
        Returns:
            Dictionary with tree metrics
        """
        if not nodes:
            return {
                "max_depth": 0.0,
                "avg_branching_factor": 0.0,
                "tree_size": 0.0,
            }
        
        depths = [get_depth(node) for node in nodes]
        max_depth = max(depths) if depths else 0
        
        # Calculate branching factors
        branching_factors = []
        for node in nodes:
            children = get_children(node)
            if children:
                branching_factors.append(len(children))
        
        avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0.0
        
        return {
            "max_depth": float(max_depth),
            "avg_branching_factor": avg_branching,
            "tree_size": float(len(nodes)),
        }


class EnhancedMCTS:
    """
    Enhanced MCTS with mathematical foundations.
    
    Implements UCB1: UCB1(v) = Q(v) + c · √(ln(N(v_parent)) / N(v))
    """
    
    @staticmethod
    def calculate_ucb1(
        node_value: float,
        node_visits: int,
        parent_visits: int,
        exploration_constant: float = math.sqrt(2)
    ) -> float:
        """
        Calculate UCB1 value: UCB1(v) = Q(v) + c · √(ln(N(v_parent)) / N(v)).
        
        Args:
            node_value: Average value Q(v)
            node_visits: Visit count N(v)
            parent_visits: Parent visit count N(v_parent)
            exploration_constant: Exploration constant c
            
        Returns:
            UCB1 value
        """
        if node_visits == 0:
            return float('inf')
        
        if parent_visits == 0:
            return node_value
        
        exploitation = node_value
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / node_visits
        )
        
        return exploitation + exploration
    
    @staticmethod
    def update_value(
        current_value: float,
        current_visits: int,
        new_value: float
    ) -> Tuple[float, int]:
        """
        Update node value: Q(v) ← (N(v) · Q(v) + V_new) / (N(v) + 1).
        
        Args:
            current_value: Current average value Q(v)
            current_visits: Current visit count N(v)
            new_value: New evaluation value V_new
            
        Returns:
            Tuple of (updated_value, updated_visits)
        """
        new_visits = current_visits + 1
        updated_value = (current_value * current_visits + new_value) / new_visits
        
        return updated_value, new_visits


@dataclass
class ThoughtNode:
    """
    Represents a node in the Tree of Thought reasoning structure.
    
    Attributes:
        id: Unique identifier for the node
        depth: Depth of the node in the tree (0 = root)
        text: Reasoning text accumulated so far
        score: Evaluator score (heuristic value)
        children: List of child ThoughtNode instances
        parent: Optional parent ThoughtNode
        is_leaf: Whether this is a leaf node (no children)
        visit_count: Number of times visited (for MCTS)
        value_sum: Sum of values from evaluations (for MCTS)
    """
    
    id: UUID
    depth: int
    text: str
    score: float = 0.0
    children: List["ThoughtNode"] = field(default_factory=list)
    parent: Optional["ThoughtNode"] = None
    is_leaf: bool = True
    visit_count: int = 0
    value_sum: float = 0.0
    
    def get_path(self) -> str:
        """
        Get the full reasoning path from root to this node.
        
        Returns:
            Complete reasoning text from root to this node
        """
        if self.parent is None:
            return self.text
        return self.parent.get_path() + self.text
    
    def get_average_value(self) -> float:
        """
        Get the average value for MCTS.
        
        Returns:
            Average value (value_sum / visit_count) or 0.0 if not visited
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class ToTConfig:
    """
    Configuration for Tree-of-Thought reasoning.
    
    Attributes:
        max_depth: Maximum depth of the reasoning tree
        branch_factor: Number of candidate thoughts to generate per node
        beam_width: Number of nodes to keep in beam search
        temperature: Sampling temperature for thought generation
        top_p: Nucleus sampling parameter
        max_thought_length: Maximum length of a single thought in tokens
        search_strategy: Search strategy (beam, bfs, dfs, mcts)
        mcts_simulations: Number of MCTS simulations (if using MCTS)
        mcts_exploration: Exploration constant for UCB1 (if using MCTS)
        evaluation_temperature: Temperature for evaluation prompts
        answer_prefix: Prefix for final answer extraction
        return_tree: Whether to return the full tree structure
    """
    
    max_depth: int = 5
    branch_factor: int = 3
    beam_width: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    max_thought_length: int = 200
    search_strategy: SearchStrategy = SearchStrategy.BEAM
    mcts_simulations: int = 100
    mcts_exploration: float = 1.414  # sqrt(2)
    evaluation_temperature: float = 0.3
    answer_prefix: str = "Final answer:"
    return_tree: bool = False


class LLMBackend:
    """
    Abstract interface for LLM backend.
    
    This defines the contract that any LLM implementation must follow
    to work with the ToT framework.
    """
    
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
            
        Returns:
            Generated text
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement generate method")


class PromptBuilder:
    """
    Builds prompts for Tree-of-Thought reasoning.
    
    Creates prompts for:
    - Initial problem statement
    - Thought expansion (generating candidate continuations)
    - Evaluation (scoring partial reasoning paths)
    - Answer extraction (extracting final answer from reasoning)
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that reasons through problems
by exploring multiple possible approaches. Consider different perspectives and evaluate
which reasoning paths are most promising."""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the PromptBuilder.
        
        Args:
            system_prompt: Custom system prompt (uses default if None)
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
    
    def build_initial_prompt(self, problem: str) -> str:
        """
        Build the initial prompt for the problem.
        
        Args:
            problem: The problem statement
            
        Returns:
            Complete initial prompt
        """
        return f"{self.system_prompt}\n\nProblem: {problem}\n\nLet's explore different approaches to solve this problem."
    
    def build_expansion_prompt(
        self,
        problem: str,
        partial_reasoning: str,
        num_branches: int,
    ) -> str:
        """
        Build a prompt for expanding a node (generating candidate thoughts).
        
        Args:
            problem: The original problem statement
            partial_reasoning: Current reasoning path so far
            num_branches: Number of candidate thoughts to generate
            
        Returns:
            Prompt for thought expansion
        """
        return f"""{self.system_prompt}

Problem: {problem}

Current reasoning:
{partial_reasoning}

Propose {num_branches} possible next steps, each continuing the reasoning differently.
Each step should explore a different approach or perspective.
Return them as numbered items:

1. ...
2. ...
3. ...
..."""
    
    def build_evaluation_prompt(
        self,
        problem: str,
        partial_reasoning: str,
    ) -> str:
        """
        Build a prompt for evaluating a partial reasoning path.
        
        Args:
            problem: The original problem statement
            partial_reasoning: The reasoning path to evaluate
            
        Returns:
            Prompt for evaluation
        """
        return f"""Given this reasoning so far, rate its plausibility and likelihood of correctness from 1 to 10.

Problem: {problem}

Reasoning:
{partial_reasoning}

Provide a single number from 1 to 10, where:
- 1-3: Very unlikely to lead to correct answer
- 4-6: Somewhat plausible but uncertain
- 7-8: Likely to be on the right track
- 9-10: Very promising and likely correct

Score:"""
    
    def build_answer_extraction_prompt(
        self,
        problem: str,
        reasoning: str,
    ) -> str:
        """
        Build a prompt for extracting the final answer from reasoning.
        
        Args:
            problem: The original problem statement
            reasoning: Complete reasoning path
            
        Returns:
            Prompt for answer extraction
        """
        return f"""{self.system_prompt}

Problem: {problem}

Reasoning:
{reasoning}

Based on the reasoning above, provide the final answer.
Format: {self.answer_prefix} [your answer]"""
    
    @property
    def answer_prefix(self) -> str:
        """Get the answer prefix."""
        return "Final answer:"


class NodeExpander:
    """
    Expands nodes by generating candidate thoughts using the LLM.
    
    Given a partial reasoning sequence, asks the LLM to propose
    multiple different next steps (thoughts).
    """
    
    def __init__(
        self,
        llm: LLMBackend,
        config: ToTConfig,
        prompt_builder: PromptBuilder,
    ):
        """
        Initialize the NodeExpander.
        
        Args:
            llm: LLM backend instance
            config: ToT configuration
            prompt_builder: PromptBuilder instance
        """
        self.llm = llm
        self.config = config
        self.prompt_builder = prompt_builder
    
    def expand(
        self,
        problem: str,
        node: ThoughtNode,
    ) -> List[str]:
        """
        Expand a node by generating candidate thoughts.
        
        Args:
            problem: The original problem statement
            node: The node to expand
            
        Returns:
            List of candidate thought strings
        """
        partial_reasoning = node.get_path()
        
        prompt = self.prompt_builder.build_expansion_prompt(
            problem=problem,
            partial_reasoning=partial_reasoning,
            num_branches=self.config.branch_factor,
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=self.config.max_thought_length * self.config.branch_factor,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=None,
            )
            
            # Parse the response to extract numbered thoughts
            thoughts = self._parse_thoughts(response)
            
            # Limit to branch_factor
            return thoughts[:self.config.branch_factor]
        
        except Exception as e:
            logger.error(f"Error expanding node: {e}")
            return []
    
    def _parse_thoughts(self, response: str) -> List[str]:
        """
        Parse numbered thoughts from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            List of thought strings
        """
        thoughts = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Match patterns like "1. thought", "1) thought", "- thought"
            if not line:
                continue
            
            # Remove leading number/bullet
            for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
                          '1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)', '10)',
                          '-', '*']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            
            if line:
                thoughts.append(line)
        
        # If no numbered format found, split by newlines
        if not thoughts:
            thoughts = [line.strip() for line in lines if line.strip()]
        
        return thoughts


class Evaluator:
    """
    Evaluates partial reasoning paths using the LLM as a heuristic.
    
    Uses the LLM to assign a score to each node's partial reasoning,
    estimating the likelihood that extending this path will yield a correct answer.
    """
    
    def __init__(
        self,
        llm: LLMBackend,
        config: ToTConfig,
        prompt_builder: PromptBuilder,
    ):
        """
        Initialize the Evaluator.
        
        Args:
            llm: LLM backend instance
            config: ToT configuration
            prompt_builder: PromptBuilder instance
        """
        self.llm = llm
        self.config = config
        self.prompt_builder = prompt_builder
    
    def evaluate(
        self,
        problem: str,
        node: ThoughtNode,
    ) -> float:
        """
        Evaluate a node's partial reasoning path.
        
        Args:
            problem: The original problem statement
            node: The node to evaluate
            
        Returns:
            Score from 1.0 to 10.0
        """
        partial_reasoning = node.get_path()
        
        prompt = self.prompt_builder.build_evaluation_prompt(
            problem=problem,
            partial_reasoning=partial_reasoning,
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=self.config.evaluation_temperature,
                top_p=0.9,
                stop=None,
            )
            
            # Extract score from response
            score = self._parse_score(response)
            return score
        
        except Exception as e:
            logger.error(f"Error evaluating node: {e}")
            return 5.0  # Default neutral score
    
    def _parse_score(self, response: str) -> float:
        """
        Parse score from evaluation response.
        
        Args:
            response: LLM response text
            
        Returns:
            Score from 1.0 to 10.0
        """
        import re
        
        # Try to find a number in the response
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if numbers:
            score = float(numbers[0])
            # Clamp to [1.0, 10.0]
            score = max(1.0, min(10.0, score))
            return score
        
        # Default if no number found
        return 5.0


class AnswerExtractor:
    """
    Extracts the final answer from a complete reasoning path.
    
    When a leaf node produces a final answer, this extracts and verifies it.
    """
    
    def __init__(
        self,
        llm: LLMBackend,
        prompt_builder: PromptBuilder,
    ):
        """
        Initialize the AnswerExtractor.
        
        Args:
            llm: LLM backend instance
            prompt_builder: PromptBuilder instance
        """
        self.llm = llm
        self.prompt_builder = prompt_builder
    
    def extract(
        self,
        problem: str,
        reasoning: str,
    ) -> str:
        """
        Extract final answer from reasoning path.
        
        Args:
            problem: The original problem statement
            reasoning: Complete reasoning path
            
        Returns:
            Extracted final answer
        """
        prompt = self.prompt_builder.build_answer_extraction_prompt(
            problem=problem,
            reasoning=reasoning,
        )
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
                top_p=0.9,
                stop=None,
            )
            
            # Extract answer after prefix
            answer = self._extract_answer(response)
            return answer
        
        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            return reasoning  # Fallback to reasoning text
    
    def _extract_answer(self, response: str) -> str:
        """
        Extract answer from response text.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted answer
        """
        prefix = self.prompt_builder.answer_prefix.lower()
        response_lower = response.lower()
        
        # Find answer prefix
        if prefix in response_lower:
            idx = response_lower.find(prefix)
            answer = response[idx + len(prefix):].strip()
            return answer
        
        # If no prefix found, return the response
        return response.strip()


class SearchController:
    """
    Implements search algorithms for exploring the reasoning tree.
    
    Supports beam search, BFS, DFS, and MCTS strategies.
    """
    
    def __init__(
        self,
        config: ToTConfig,
        expander: NodeExpander,
        evaluator: Evaluator,
    ):
        """
        Initialize the SearchController.
        
        Args:
            config: ToT configuration
            expander: NodeExpander instance
            evaluator: Evaluator instance
        """
        self.config = config
        self.expander = expander
        self.evaluator = evaluator
    
    def search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
        """
        Search the reasoning tree to find the best leaf nodes.
        
        Args:
            problem: The original problem statement
            root: Root node of the tree
            
        Returns:
            List of best leaf nodes
        """
        if self.config.search_strategy == SearchStrategy.BEAM:
            return self._beam_search(problem, root)
        elif self.config.search_strategy == SearchStrategy.BFS:
            return self._bfs_search(problem, root)
        elif self.config.search_strategy == SearchStrategy.DFS:
            return self._dfs_search(problem, root)
        elif self.config.search_strategy == SearchStrategy.MCTS:
            return self._mcts_search(problem, root)
        elif self.config.search_strategy == SearchStrategy.QUANTUM:
            return self._quantum_search(problem, root)
        else:
            logger.warning(f"Unknown search strategy: {self.config.search_strategy}, using beam search")
            return self._beam_search(problem, root)
    
    def _beam_search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
        """
        Perform beam search on the reasoning tree.
        
        Args:
            problem: The original problem statement
            root: Root node of the tree
            
        Returns:
            List of best leaf nodes
        """
        frontier = [root]
        
        for depth in range(self.config.max_depth):
            if not frontier:
                break
            
            new_frontier = []
            
            # Expand all nodes in current frontier
            for node in frontier:
                if node.depth >= self.config.max_depth:
                    continue
                
                # Generate candidate thoughts
                thoughts = self.expander.expand(problem, node)
                
                # Create child nodes
                for thought in thoughts:
                    child = ThoughtNode(
                        id=uuid4(),
                        depth=node.depth + 1,
                        text=thought,
                        parent=node,
                    )
                    
                    # Evaluate the child node
                    child.score = self.evaluator.evaluate(problem, child)
                    
                    node.children.append(child)
                    node.is_leaf = False
                    new_frontier.append(child)
            
            # Keep only top-scoring nodes (beam search)
            # Enhanced scoring: score(v) = α·heuristic + β·depth_penalty + γ·path_prob
            if new_frontier:
                # Calculate path probabilities for enhanced scoring
                for node in new_frontier:
                    # Get path scores (node scores along path)
                    path_scores = []
                    current = node
                    while current is not None:
                        path_scores.append(current.score / 10.0)  # Normalize to [0, 1]
                        current = current.parent
                    
                    # Calculate path probability
                    path_prob = TreeGraphTheory.calculate_path_probability(
                        path_scores, normalize=True
                    )
                    
                    # Enhanced score: score = heuristic + depth_penalty + path_prob
                    depth_penalty = -0.1 * node.depth  # λ = 0.1
                    enhanced_score = node.score + depth_penalty + 10.0 * path_prob
                    node.score = enhanced_score
                
                new_frontier.sort(key=lambda n: n.score, reverse=True)
                frontier = new_frontier[:self.config.beam_width]
            else:
                break
        
        return frontier
    
    def _bfs_search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
        """
        Perform breadth-first search on the reasoning tree.
        
        Args:
            problem: The original problem statement
            root: Root node of the tree
            
        Returns:
            List of leaf nodes at max depth
        """
        queue = deque([root])
        leaves = []
        
        while queue:
            node = queue.popleft()
            
            if node.depth >= self.config.max_depth:
                leaves.append(node)
                continue
            
            # Expand node
            thoughts = self.expander.expand(problem, node)
            
            for thought in thoughts:
                child = ThoughtNode(
                    id=uuid4(),
                    depth=node.depth + 1,
                    text=thought,
                    parent=node,
                )
                
                child.score = self.evaluator.evaluate(problem, child)
                node.children.append(child)
                node.is_leaf = False
                queue.append(child)
        
        return leaves if leaves else [root]
    
    def _dfs_search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
        """
        Perform depth-first search on the reasoning tree.
        
        Args:
            problem: The original problem statement
            root: Root node of the tree
            
        Returns:
            List of best leaf nodes
        """
        stack = [root]
        best_leaves = []
        
        while stack:
            node = stack.pop()
            
            if node.depth >= self.config.max_depth:
                best_leaves.append(node)
                continue
            
            # Expand node
            thoughts = self.expander.expand(problem, node)
            
            # Sort by score and add to stack (best first)
            children = []
            for thought in thoughts:
                child = ThoughtNode(
                    id=uuid4(),
                    depth=node.depth + 1,
                    text=thought,
                    parent=node,
                )
                
                child.score = self.evaluator.evaluate(problem, child)
                node.children.append(child)
                node.is_leaf = False
                children.append(child)
            
            # Add children to stack in reverse order (best first)
            children.sort(key=lambda n: n.score, reverse=True)
            stack.extend(reversed(children))
        
        # Return best leaves sorted by score
        best_leaves.sort(key=lambda n: n.score, reverse=True)
        return best_leaves[:self.config.beam_width] if best_leaves else [root]
    
    def _mcts_search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
        """
        Perform Monte-Carlo Tree Search on the reasoning tree.
        
        Args:
            problem: The original problem statement
            root: Root node of the tree
            
        Returns:
            List of best leaf nodes
        """
        for _ in range(self.config.mcts_simulations):
            # Selection: traverse to leaf using UCB1
            node = self._mcts_select(root)
            
            # Expansion: expand if not at max depth
            if node.depth < self.config.max_depth and not node.children:
                thoughts = self.expander.expand(problem, node)
                for thought in thoughts:
                    child = ThoughtNode(
                        id=uuid4(),
                        depth=node.depth + 1,
                        text=thought,
                        parent=node,
                    )
                    child.score = self.evaluator.evaluate(problem, child)
                    node.children.append(child)
                    node.is_leaf = False
                
                if node.children:
                    node = node.children[0]  # Use first child for evaluation
            
            # Evaluation: evaluate the node
            value = self.evaluator.evaluate(problem, node)
            
            # Backpropagation: update values up the tree
            self._mcts_backpropagate(node, value)
        
        # Return best leaves
        return self._mcts_get_best_leaves(root)
    
    def _mcts_select(self, node: ThoughtNode) -> ThoughtNode:
        """
        Select a node using UCB1 formula.
        
        Args:
            node: Current node
            
        Returns:
            Selected node
        """
        while node.children:
            if not all(child.visit_count > 0 for child in node.children):
                # Select unvisited child
                for child in node.children:
                    if child.visit_count == 0:
                        return child
            
            # Use UCB1 to select best child (using EnhancedMCTS)
            best_child = None
            best_ucb = float('-inf')
            
            for child in node.children:
                if child.visit_count == 0:
                    ucb = float('inf')
                else:
                    ucb = EnhancedMCTS.calculate_ucb1(
                        node_value=child.get_average_value(),
                        node_visits=child.visit_count,
                        parent_visits=node.visit_count,
                        exploration_constant=self.config.mcts_exploration
                    )
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            
            node = best_child
        
        return node
    
    def _mcts_backpropagate(
        self,
        node: ThoughtNode,
        value: float,
    ) -> None:
        """
        Backpropagate value up the tree.
        
        Uses: Q(v) ← (N(v) · Q(v) + V_new) / (N(v) + 1)
        
        Args:
            node: Node to start backpropagation from
            value: Value to backpropagate
        """
        while node is not None:
            # Update using EnhancedMCTS formula
            updated_value, updated_visits = EnhancedMCTS.update_value(
                current_value=node.get_average_value(),
                current_visits=node.visit_count,
                new_value=value
            )
            node.value_sum = updated_value * updated_visits
            node.visit_count = updated_visits
            node = node.parent
    
    def _mcts_get_best_leaves(self, root: ThoughtNode) -> List[ThoughtNode]:
        """
        Get best leaf nodes from MCTS tree.
        
        Args:
            root: Root of the tree
            
        Returns:
            List of best leaf nodes
        """
        leaves = []
        stack = [root]
        
        while stack:
            node = stack.pop()
            if node.is_leaf or not node.children:
                leaves.append(node)
            else:
                stack.extend(node.children)
        
        # Sort by average value
        leaves.sort(key=lambda n: n.get_average_value(), reverse=True)
        return leaves[:self.config.beam_width] if leaves else [root]
    
    def _quantum_search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
        """
        Perform quantum-inspired tree search with superposition of paths.
        
        Implements: |ψ_tree⟩ = Σ_path α_path |path⟩
        Measurement: P(y | x) = |⟨y | ψ_tree⟩|²
        
        Args:
            problem: The original problem statement
            root: Root node of the tree
            
        Returns:
            List of best leaf nodes (sampled from quantum measurement)
        """
        # Build tree using beam search first
        leaves = self._beam_search(problem, root)
        
        if not leaves:
            return [root]
        
        # Extract paths and calculate probabilities
        paths = []
        path_probs = []
        
        for leaf in leaves:
            path = []
            current = leaf
            while current is not None:
                path.insert(0, current.text)
                current = current.parent
            
            paths.append(path)
            # Use score as probability (normalized)
            path_probs.append(max(0.0, leaf.score / 10.0))
        
        # Normalize probabilities
        total_prob = sum(path_probs)
        if total_prob > 0:
            path_probs = [p / total_prob for p in path_probs]
        else:
            path_probs = [1.0 / len(leaves)] * len(leaves)
        
        # Quantum sampling: sample leaves based on quantum amplitudes
        sampled_leaves = QuantumTreeSearch.quantum_tree_sampling(
            nodes=leaves,
            path_probs=path_probs,
            num_samples=min(self.config.beam_width, len(leaves))
        )
        
        return sampled_leaves


class ToTReasoner:
    """
    Main Tree-of-Thought reasoning engine.
    
    Implements the core ToT algorithm:
    1. Build initial prompt
    2. Create root node
    3. Search the tree using selected strategy
    4. Extract final answer from best leaf
    """
    
    def __init__(
        self,
        llm: LLMBackend,
        config: Optional[ToTConfig] = None,
    ):
        """
        Initialize the ToTReasoner.
        
        Args:
            llm: LLM backend instance
            config: ToT configuration (uses defaults if None)
        """
        self.llm = llm
        self.config = config or ToTConfig()
        
        # Initialize components
        self.prompt_builder = PromptBuilder()
        self.expander = NodeExpander(
            llm=llm,
            config=self.config,
            prompt_builder=self.prompt_builder,
        )
        self.evaluator = Evaluator(
            llm=llm,
            config=self.config,
            prompt_builder=self.prompt_builder,
        )
        self.search_controller = SearchController(
            config=self.config,
            expander=self.expander,
            evaluator=self.evaluator,
        )
        self.answer_extractor = AnswerExtractor(
            llm=llm,
            prompt_builder=self.prompt_builder,
        )
    
    def solve(
        self,
        problem: str,
    ) -> Dict[str, Any]:
        """
        Solve a problem using Tree-of-Thought reasoning.
        
        Args:
            problem: Problem statement to solve
            
        Returns:
            Dictionary with final_answer, reasoning, score, and optionally tree
        """
        logger.info(f"Starting ToT reasoning for problem: {problem[:100]}...")
        
        # Create root node
        root = ThoughtNode(
            id=uuid4(),
            depth=0,
            text="",
        )
        
        # Search the tree
        best_leaves = self.search_controller.search(problem, root)
        
        if not best_leaves:
            logger.warning("No leaves found in search, using root")
            best_leaves = [root]
        
        # Select best leaf (with enhanced path selection)
        if self.config.search_strategy == SearchStrategy.QUANTUM:
            # Use quantum measurement for answer selection
            paths = [leaf.get_path().split("\n") for leaf in best_leaves]
            answers = [self.answer_extractor.extract(problem, leaf.get_path()) for leaf in best_leaves]
            path_probs = [max(0.0, leaf.score / 10.0) for leaf in best_leaves]
            
            # Normalize probabilities
            total_prob = sum(path_probs)
            if total_prob > 0:
                path_probs = [p / total_prob for p in path_probs]
            else:
                path_probs = [1.0 / len(best_leaves)] * len(best_leaves)
            
            # Quantum measurement
            final_answer, confidence = QuantumTreeSearch.quantum_measurement(
                paths=paths,
                answers=answers,
                path_probs=path_probs
            )
            
            # Find corresponding leaf
            best_leaf = best_leaves[0]  # Default
            for i, answer in enumerate(answers):
                if answer.lower().strip() == final_answer.lower().strip():
                    best_leaf = best_leaves[i]
                    break
        else:
            # Standard selection: find optimal path
            paths = [[leaf.get_path()] for leaf in best_leaves]
            path_scores = [leaf.score for leaf in best_leaves]
            path_lengths = [leaf.depth for leaf in best_leaves]
            
            optimal_path = TreeGraphTheory.find_optimal_path(
                paths=paths,
                path_scores=path_scores,
                path_lengths=path_lengths,
                lambda_reg=0.1
            )
            
            # Find corresponding leaf
            best_leaf = best_leaves[0]
            if optimal_path:
                for i, path in enumerate(paths):
                    if path == optimal_path:
                        best_leaf = best_leaves[i]
                        break
            
            confidence = best_leaf.score / 10.0
        
        # Extract final answer
        reasoning = best_leaf.get_path()
        if self.config.search_strategy != SearchStrategy.QUANTUM:
            final_answer = self.answer_extractor.extract(problem, reasoning)
        
        # Calculate tree metrics
        all_nodes = []
        stack = [root]
        while stack:
            node = stack.pop()
            all_nodes.append(node)
            stack.extend(node.children)
        
        tree_metrics = TreeGraphTheory.calculate_tree_metrics(
            nodes=all_nodes,
            get_depth=lambda n: n.depth,
            get_children=lambda n: n.children
        )
        
        # Calculate path entropy and diversity
        leaf_paths = []
        for leaf in best_leaves:
            path = []
            current = leaf
            while current is not None:
                path.insert(0, current.text)
                current = current.parent
            leaf_paths.append(path)
        
        path_probs_for_entropy = [max(0.0, leaf.score / 10.0) for leaf in best_leaves]
        total_prob = sum(path_probs_for_entropy)
        if total_prob > 0:
            path_probs_for_entropy = [p / total_prob for p in path_probs_for_entropy]
        
        path_entropy = TreeInformationTheory.path_entropy(path_probs_for_entropy)
        tree_diversity = TreeInformationTheory.tree_diversity(leaf_paths)
        
        # Calculate energy-based metrics if scores available
        energy_metrics = {}
        if best_leaves:
            path_logprobs = [math.log(max(0.001, leaf.score / 10.0)) for leaf in best_leaves]
            path_energies = [TreeEnergyFunction.calculate_path_energy(logprob) for logprob in path_logprobs]
            partition_func = TreeEnergyFunction.tree_partition_function(
                path_energies, self.config.temperature
            )
            free_energy = TreeEnergyFunction.tree_free_energy(partition_func, self.config.temperature)
            
            energy_metrics = {
                "partition_function": partition_func,
                "free_energy": free_energy,
            }
        
        result = {
            "final_answer": final_answer,
            "reasoning": reasoning,
            "score": best_leaf.score,
            "depth": best_leaf.depth,
            "confidence": confidence,
            "tree_metrics": tree_metrics,
            "path_entropy": path_entropy,
            "tree_diversity": tree_diversity,
            **energy_metrics,
        }
        
        if self.config.return_tree:
            result["tree"] = root
        
        logger.info(f"ToT reasoning completed. Score: {best_leaf.score:.2f}, Confidence: {confidence:.2f}")
        
        return result


class AgentLLMAdapter(LLMBackend):
    """
    Adapter to use Agent's LLM with the ToT framework.
    
    Wraps the Agent's LLM interface to match the LLMBackend contract.
    """
    
    def __init__(self, agent: Any):
        """
        Initialize the adapter.
        
        Args:
            agent: Agent instance with an LLM, or direct LLM instance/callable
        """
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
        """
        Generate text using the Agent's LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
            
        Returns:
            Generated text
        """
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
            logger.error(f"Error in AgentLLMAdapter.generate: {e}")
            # Fallback to agent's run method
            if hasattr(self.agent, 'run'):
                try:
                    return str(self.agent.run(prompt))
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    raise
            raise


class ToTAgent:
    """
    Tree-of-Thought Agent for exploring multiple reasoning paths.
    
    This agent implements the Tree-of-Thought (ToT) reasoning framework,
    which introduces a tree-structured latent variable R representing
    multiple candidate reasoning paths and searches over this tree.
    
    Mathematical Foundation:
        p_θ(y | x) = Σ_{R ∈ T} p_θ(R | x) · p_θ(y | R, x)
        
        Where:
        - x = input (question, task description)
        - y = final answer
        - R = {r^(1), r^(2), ..., r^(k)} = set of candidate reasoning paths
        - T = set of reasoning trees
    
    Attributes:
        agent_name: Name of the agent
        model_name: LLM model to use
        config: ToT configuration
        reasoner: Internal ToTReasoner instance
    
    Example:
        >>> from swarms.agents import ToTAgent
        >>> # Using model_name
        >>> agent = ToTAgent(
        ...     agent_name="tot-agent",
        ...     model_name="gpt-4o",
        ... )
        >>> result = agent.run("Solve: If a train travels 120 miles in 2 hours, what is its average speed?")
        >>> print(result)
        
        >>> # Using llm directly (any LLM instance with a 'run' method)
        >>> from swarms import LiteLLM
        >>> llm = LiteLLM(model_name="gpt-4o")
        >>> agent = ToTAgent(llm=llm)
        >>> result = agent.run("Your problem here")
    """
    
    def __init__(
        self,
        agent_name: str = "tot-agent",
        model_name: Optional[str] = "gpt-4o",
        llm: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        config: Optional[ToTConfig] = None,
        agent: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the ToTAgent.
        
        Args:
            agent_name: Name of the agent
            model_name: LLM model name (used if agent/llm not provided)
            llm: Optional LLM instance or callable (takes precedence over model_name)
            system_prompt: Optional custom system prompt
            config: ToT configuration (uses defaults if None)
            agent: Optional Agent instance to use (if provided, uses its LLM)
            **kwargs: Additional arguments passed to Agent if creating one
        """
        self.agent_name = agent_name
        self.model_name = model_name
        self.config = config or ToTConfig()
        
        # Priority: agent > llm > model_name
        if agent is not None:
            # Use provided Agent instance
            self.agent = agent
            llm_adapter = AgentLLMAdapter(agent)
        elif llm is not None:
            # Use provided LLM directly (can be callable or LLM instance)
            self.agent = llm
            llm_adapter = AgentLLMAdapter(llm)
        else:
            # Create Agent from model_name
            if model_name is None:
                raise ValueError("Either 'agent', 'llm', or 'model_name' must be provided")
            
            # Import Agent here to avoid circular imports
            from swarms.structs.agent import Agent
            
            self.agent = Agent(
                agent_name=agent_name,
                model_name=model_name,
                system_prompt=system_prompt,
                **kwargs,
            )
            llm_adapter = AgentLLMAdapter(self.agent)
        
        # Initialize the ToT reasoner
        self.reasoner = ToTReasoner(
            llm=llm_adapter,
            config=self.config,
        )
    
    def run(
        self,
        task: str,
        return_tree: Optional[bool] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Run the Tree-of-Thought agent on a task.
        
        Args:
            task: Task or question to solve
            return_tree: Whether to return full result with tree (defaults to config setting)
            
        Returns:
            Final answer string, or full result dict if return_tree=True
        """
        # Temporarily override return_tree if specified
        original_return_tree = self.config.return_tree
        if return_tree is not None:
            self.config.return_tree = return_tree
        
        try:
            result = self.reasoner.solve(task)
            
            # Return based on configuration
            if self.config.return_tree:
                return result
            else:
                return result["final_answer"]
        finally:
            # Restore original setting
            self.config.return_tree = original_return_tree


__all__ = [
    "ToTAgent",
]

