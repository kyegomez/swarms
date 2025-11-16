"""
Chain-of-Thought (CoT) Reasoning Framework Implementation.

This module implements a comprehensive Chain-of-Thought reasoning system based on
the formal framework where we introduce an explicit latent sequence of reasoning
tokens between input and output, and search over that latent space with a sequence model.

Mathematical Foundation:
    Core Probabilistic Model:
        p_θ(y, r | x) = p_θ(r | x) · p_θ(y | x, r)
    
    Where:
    - x = input (question, task description) ∈ X
    - y = final answer ∈ Y
    - r = (r₁, ..., r_T) = reasoning trace (CoT), a sequence of tokens
    - R = latent reasoning variable (reasoning space)
    - θ = model parameters
    
    Variational Lower Bound (ELBO):
        log p_θ(y | x) ≥ E_{q_φ(r|x,y)}[log p_θ(y | x, r)] - KL(q_φ(r|x,y) || p_θ(r|x))
        
        Where q_φ(r|x,y) is the variational posterior approximating the true posterior.
    
    Information-Theoretic Formulation:
        I(X; Y | R) = H(Y | R) - H(Y | X, R)
        
        Mutual information between input X and output Y given reasoning R.
        
        Entropy of reasoning trace:
            H(R | X) = -Σ_{r} p_θ(r | x) log p_θ(r | x)
        
        Conditional entropy of answer:
            H(Y | X, R) = -Σ_{y,r} p_θ(y, r | x) log p_θ(y | x, r)
    
    Quantum-Inspired Superposition of Reasoning Paths:
        |ψ⟩ = Σ_{r} α_r |r⟩ ⊗ |y_r⟩
        
        Where:
        - |ψ⟩ = quantum state representing superposition of reasoning paths
        - α_r = amplitude for reasoning trace r: α_r = √(p_θ(r | x))
        - |r⟩ = basis state for reasoning trace r
        - |y_r⟩ = answer state conditioned on r
        
        Measurement probability:
            P(y | x) = |⟨y | ψ⟩|² = |Σ_{r: y_r=y} α_r|²
    
    Graph-Theoretic Reasoning Representation:
        G = (V, E) where:
        - V = {v₁, ..., v_T} = reasoning steps (vertices)
        - E = {(v_i, v_j) | v_i → v_j} = causal dependencies (edges)
        
        Path probability:
            P(path) = Π_{(v_i,v_j)∈path} P(v_j | v_i, x)
        
        Shortest reasoning path (Dijkstra-like):
            r* = argmin_{r} -log p_θ(r | x) + λ·L(r)
            
            Where L(r) is the length penalty and λ is regularization.
    
    Statistical Mechanics (Boltzmann Distribution):
        p_θ(r | x) = (1/Z(x)) exp(-E_θ(r, x) / T)
        
        Where:
        - E_θ(r, x) = energy function (negative log-likelihood)
        - T = temperature parameter (controls exploration)
        - Z(x) = partition function: Z(x) = Σ_{r} exp(-E_θ(r, x) / T)
        
        Free energy:
            F(x) = -T log Z(x) = -T log Σ_{r} exp(-E_θ(r, x) / T)
    
    Self-Consistency (Ensemble Aggregation):
        Marginalized answer distribution:
            p(y | x) = Σ_{r} p_θ(r | x) · p_θ(y | x, r)
        
        Majority voting with weights:
            ŷ = argmax_{y} Σ_{i=1}^N w_i · 𝟙[y_i = y]
            
            Where w_i = p_θ(r_i | x) or w_i = score(r_i) from verifier.
        
        Confidence via entropy:
            Confidence = 1 - (H(Y | X) / log |Y|)
            
            Where H(Y | X) = -Σ_{y} p(y | x) log p(y | x)
    
    Optimization (Variational Inference):
        Objective:
            L(θ, φ) = E_{q_φ(r|x,y)}[log p_θ(y | x, r)] - β·KL(q_φ(r|x,y) || p_θ(r|x))
        
        Gradient:
            ∇_θ L = E_{q_φ}[∇_θ log p_θ(y | x, r)]
            ∇_φ L = E_{q_φ}[log p_θ(y | x, r) · ∇_φ log q_φ(r|x,y)] - β·∇_φ KL(q_φ || p_θ)
    
    Computational Complexity:
        Time complexity: O(T · |V| · d) where:
        - T = max reasoning length
        - |V| = vocabulary size
        - d = model dimension
        
        Space complexity: O(T · d) for storing reasoning trace.
        
        With self-consistency (N samples): O(N · T · |V| · d)

At inference time:
    1. Sample or search for a plausible reasoning trace r* from p_θ(r | x)
    2. Decode y from p_θ(y | x, r*)
    
    Search strategies:
    - Greedy: r* = argmax_{r} p_θ(r | x)
    - Sampling: r* ~ p_θ(r | x) (Boltzmann sampling)
    - Beam search: Top-K reasoning paths
    - Quantum-inspired: Sample from |ψ⟩ = Σ_{r} α_r |r⟩
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import re
from collections import Counter
import math
import random

from loguru import logger


class DecodingStrategy(str, Enum):
    """Decoding strategies for generating reasoning traces."""
    
    GREEDY = "greedy"
    SAMPLING = "sampling"
    NUCLEUS = "nucleus"
    QUANTUM = "quantum"  # Quantum-inspired superposition sampling


class InformationTheory:
    """
    Information-theoretic utilities for reasoning analysis.
    
    Implements entropy, mutual information, and related measures.
    """
    
    @staticmethod
    def entropy(probabilities: List[float]) -> float:
        """
        Calculate Shannon entropy: H(X) = -Σ p(x) log p(x).
        
        Args:
            probabilities: List of probabilities (should sum to ~1)
            
        Returns:
            Entropy value in bits
        """
        if not probabilities:
            return 0.0
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            return 0.0
        
        normalized = [p / total for p in probabilities]
        
        # Calculate entropy
        h = 0.0
        for p in normalized:
            if p > 0:
                h -= p * math.log2(p)
        
        return h
    
    @staticmethod
    def conditional_entropy(
        joint_probs: Dict[Tuple[str, str], float],
        marginal_probs: Dict[str, float]
    ) -> float:
        """
        Calculate conditional entropy: H(Y | X) = -Σ p(x,y) log p(y|x).
        
        Args:
            joint_probs: Dictionary mapping (x, y) to joint probability
            marginal_probs: Dictionary mapping x to marginal probability
            
        Returns:
            Conditional entropy value
        """
        h = 0.0
        for (x, y), p_xy in joint_probs.items():
            p_x = marginal_probs.get(x, 0.0)
            if p_x > 0 and p_xy > 0:
                p_y_given_x = p_xy / p_x
                h -= p_xy * math.log2(p_y_given_x)
        
        return h
    
    @staticmethod
    def mutual_information(
        joint_probs: Dict[Tuple[str, str], float],
        x_marginal: Dict[str, float],
        y_marginal: Dict[str, float]
    ) -> float:
        """
        Calculate mutual information: I(X; Y) = H(Y) - H(Y | X).
        
        Args:
            joint_probs: Dictionary mapping (x, y) to joint probability
            x_marginal: Dictionary mapping x to marginal probability
            y_marginal: Dictionary mapping y to marginal probability
            
        Returns:
            Mutual information value
        """
        # H(Y)
        h_y = InformationTheory.entropy(list(y_marginal.values()))
        
        # H(Y | X)
        h_y_given_x = InformationTheory.conditional_entropy(joint_probs, x_marginal)
        
        return h_y - h_y_given_x
    
    @staticmethod
    def calculate_trace_entropy(traces: List[CoTTrace]) -> float:
        """
        Calculate entropy of reasoning traces: H(R | X).
        
        Args:
            traces: List of reasoning traces
            
        Returns:
            Entropy value
        """
        if not traces:
            return 0.0
        
        # Extract unique trace signatures (simplified: use step count and first few words)
        trace_signatures = []
        for trace in traces:
            sig = f"{len(trace.steps)}:{trace.steps[0].text[:20] if trace.steps else ''}"
            trace_signatures.append(sig)
        
        # Count frequencies
        counts = Counter(trace_signatures)
        total = len(trace_signatures)
        probs = [count / total for count in counts.values()]
        
        return InformationTheory.entropy(probs)


class QuantumSampler:
    """
    Quantum-inspired sampling for reasoning paths.
    
    Implements superposition-based sampling: |ψ⟩ = Σ_r α_r |r⟩
    """
    
    @staticmethod
    def calculate_amplitudes(probabilities: List[float]) -> List[float]:
        """
        Calculate quantum amplitudes: α_r = √(p_θ(r | x)).
        
        Args:
            probabilities: List of probabilities
            
        Returns:
            List of amplitudes
        """
        return [math.sqrt(max(0.0, p)) for p in probabilities]
    
    @staticmethod
    def measure_state(
        traces: List[CoTTrace],
        answers: List[str],
        probabilities: Optional[List[float]] = None
    ) -> Tuple[str, float]:
        """
        Quantum measurement: P(y | x) = |⟨y | ψ⟩|² = |Σ_{r: y_r=y} α_r|².
        
        Args:
            traces: List of reasoning traces
            answers: List of answers corresponding to traces
            probabilities: Optional probabilities for traces (uniform if None)
            
        Returns:
            Tuple of (most likely answer, probability)
        """
        if not traces or not answers:
            return "", 0.0
        
        if probabilities is None:
            probabilities = [1.0 / len(traces)] * len(traces)
        
        # Calculate amplitudes
        amplitudes = QuantumSampler.calculate_amplitudes(probabilities)
        
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
    def quantum_sampling(
        traces: List[CoTTrace],
        probabilities: List[float],
        num_samples: int = 1
    ) -> List[CoTTrace]:
        """
        Sample traces using quantum-inspired superposition.
        
        Args:
            traces: List of reasoning traces
            probabilities: Probabilities for each trace
            num_samples: Number of samples to generate
            
        Returns:
            List of sampled traces
        """
        if not traces:
            return []
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            probabilities = [1.0 / len(traces)] * len(traces)
        else:
            probabilities = [p / total for p in probabilities]
        
        # Calculate amplitudes
        amplitudes = QuantumSampler.calculate_amplitudes(probabilities)
        
        # Sample based on amplitude squared (measurement probability)
        probs = [amp ** 2 for amp in amplitudes]
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
        
        # Sample
        sampled_indices = random.choices(
            range(len(traces)),
            weights=probs,
            k=num_samples
        )
        
        return [traces[i] for i in sampled_indices]


class EnergyFunction:
    """
    Energy-based functions for statistical mechanics formulation.
    
    Implements: E(r, x) = -log p_θ(r | x) and Boltzmann distribution.
    """
    
    @staticmethod
    def calculate_energy(logprob: float) -> float:
        """
        Calculate energy: E(r, x) = -log p_θ(r | x).
        
        Args:
            logprob: Log probability of reasoning trace
            
        Returns:
            Energy value
        """
        return -logprob
    
    @staticmethod
    def boltzmann_weight(energy: float, temperature: float) -> float:
        """
        Calculate Boltzmann weight: w(r) = exp(-E(r, x) / T).
        
        Args:
            energy: Energy value E(r, x)
            temperature: Temperature parameter T
            
        Returns:
            Boltzmann weight
        """
        if temperature <= 0:
            return 0.0 if energy > 0 else 1.0
        
        return math.exp(-energy / temperature)
    
    @staticmethod
    def partition_function(energies: List[float], temperature: float) -> float:
        """
        Calculate partition function: Z(x) = Σ_r exp(-E_θ(r, x) / T).
        
        Args:
            energies: List of energy values
            temperature: Temperature parameter T
            
        Returns:
            Partition function value
        """
        if temperature <= 0:
            return 1.0
        
        weights = [EnergyFunction.boltzmann_weight(e, temperature) for e in energies]
        return sum(weights)
    
    @staticmethod
    def free_energy(partition_function: float, temperature: float) -> float:
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
    def boltzmann_sampling(
        traces: List[CoTTrace],
        temperature: float,
        num_samples: int = 1
    ) -> List[CoTTrace]:
        """
        Sample traces using Boltzmann distribution.
        
        Args:
            traces: List of reasoning traces
            temperature: Temperature parameter T
            num_samples: Number of samples to generate
            
        Returns:
            List of sampled traces
        """
        if not traces:
            return []
        
        # Calculate energies
        energies = [EnergyFunction.calculate_energy(trace.logprob) for trace in traces]
        
        # Calculate partition function
        z = EnergyFunction.partition_function(energies, temperature)
        
        if z <= 0:
            # Fallback to uniform sampling
            return random.sample(traces, min(num_samples, len(traces)))
        
        # Calculate Boltzmann weights
        weights = [
            EnergyFunction.boltzmann_weight(e, temperature) / z
            for e in energies
        ]
        
        # Sample
        sampled_indices = random.choices(
            range(len(traces)),
            weights=weights,
            k=num_samples
        )
        
        return [traces[i] for i in sampled_indices]


class GraphReasoning:
    """
    Graph-theoretic representation and path finding for reasoning.
    
    Implements reasoning as a graph G = (V, E) with path probabilities.
    """
    
    @staticmethod
    def build_reasoning_graph(trace: CoTTrace) -> Dict[int, List[int]]:
        """
        Build graph representation: G = (V, E) from reasoning trace.
        
        Args:
            trace: Reasoning trace
            
        Returns:
            Adjacency list representation of the graph
        """
        graph: Dict[int, List[int]] = {}
        
        for i, step in enumerate(trace.steps):
            if i == 0:
                graph[i] = []
            else:
                # Each step depends on previous step (linear chain)
                graph[i] = [i - 1]
                if i - 1 not in graph:
                    graph[i - 1] = []
        
        return graph
    
    @staticmethod
    def calculate_path_probability(
        trace: CoTTrace,
        step_probs: Optional[List[float]] = None
    ) -> float:
        """
        Calculate path probability: P(path) = Π_{(v_i,v_j)∈path} P(v_j | v_i, x).
        
        Args:
            trace: Reasoning trace
            step_probs: Optional probabilities for each step transition
            
        Returns:
            Path probability
        """
        if not trace.steps:
            return 0.0
        
        if step_probs is None:
            # Use uniform probabilities as default
            step_probs = [1.0] * len(trace.steps)
        
        # Product of step probabilities
        path_prob = 1.0
        for prob in step_probs:
            path_prob *= max(0.0, min(1.0, prob))
        
        return path_prob
    
    @staticmethod
    def find_shortest_path(
        traces: List[CoTTrace],
        lambda_reg: float = 0.1
    ) -> Optional[CoTTrace]:
        """
        Find shortest reasoning path: r* = argmin_r [-log p_θ(r | x) + λ·L(r)].
        
        Args:
            traces: List of reasoning traces
            lambda_reg: Regularization parameter λ
            
        Returns:
            Trace with minimum cost, or None if empty
        """
        if not traces:
            return None
        
        best_trace = None
        best_cost = float('inf')
        
        for trace in traces:
            # Cost = -log prob + λ * length
            energy = EnergyFunction.calculate_energy(trace.logprob)
            length = len(trace.steps)
            cost = energy + lambda_reg * length
            
            if cost < best_cost:
                best_cost = cost
                best_trace = trace
        
        return best_trace


@dataclass
class Question:
    """
    Represents an input question or task.
    
    Attributes:
        id: Unique identifier for the question
        text: The question text
        metadata: Additional metadata about the question
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoTStep:
    """
    Represents a single step in a chain of thought reasoning trace.
    
    Mathematical representation:
        Step t: r_t = f_θ(r_{t-1}, x, h_t)
        
        Where:
        - r_t = reasoning state at step t
        - f_θ = transition function parameterized by θ
        - h_t = hidden state: h_t = LSTM(r_{t-1}, x) or Transformer(r_{1:t-1}, x)
        - x = input context
    
    Quantum state representation:
        |r_t⟩ = U_t |r_{t-1}⟩ ⊗ |x⟩
        
        Where U_t is a unitary operator representing the reasoning transformation.
    
    Information gain:
        I(r_t; Y | r_{1:t-1}, X) = H(Y | r_{1:t-1}, X) - H(Y | r_{1:t}, X)
        
        Measures how much information step t provides about the answer.
    
    Attributes:
        index: Step index in the reasoning chain (t ∈ {1, ..., T})
        text: The reasoning text for this step (r_t)
        action: Optional tool action (for ReAct-style CoT)
        observation: Optional observation from tool execution
    """
    
    index: int
    text: str
    action: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


@dataclass
class CoTTrace:
    """
    Represents a complete reasoning trace.
    
    Mathematical representation:
        Trace: r = (r₁, r₂, ..., r_T)
        
        Joint probability:
            p_θ(r | x) = Π_{t=1}^T p_θ(r_t | r_{1:t-1}, x)
        
        Log-likelihood:
            log p_θ(r | x) = Σ_{t=1}^T log p_θ(r_t | r_{1:t-1}, x)
    
    Quantum superposition:
        |r⟩ = |r₁⟩ ⊗ |r₂⟩ ⊗ ... ⊗ |r_T⟩
        
        Entangled state:
            |ψ_trace⟩ = Σ_{r} α_r |r⟩ where α_r = √(p_θ(r | x))
    
    Energy function (Statistical Mechanics):
        E(r, x) = -log p_θ(r | x) = -Σ_{t=1}^T log p_θ(r_t | r_{1:t-1}, x)
        
        Boltzmann weight:
            w(r) = exp(-E(r, x) / T) / Z(x)
    
    Path integral formulation:
        P(r | x) = ∫ D[r(t)] exp(-S[r(t)]) / Z
        
        Where S[r(t)] is the action functional over the reasoning path.
    
    Attributes:
        steps: List of reasoning steps (r₁, ..., r_T)
        raw_text: Raw text output from the model
        logprob: Log probability of the trace: log p_θ(r | x)
        score: Optional quality score from TraceEvaluator (energy-based or learned)
    """
    
    steps: List[CoTStep]
    raw_text: str
    logprob: float = 0.0
    score: Optional[float] = None


@dataclass
class CoTResult:
    """
    Final result of CoT reasoning.
    
    Attributes:
        question: The original question
        traces: List of reasoning traces (multiple for self-consistency)
        final_answer: The final answer
        confidence: Confidence score (0-1)
        extra_metrics: Additional metrics
    """
    
    question: Question
    traces: List[CoTTrace]
    final_answer: str
    confidence: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoTConfig:
    """
    Configuration for Chain-of-Thought reasoning.
    
    Mathematical parameters:
        Temperature (T):
            p_θ(r | x) = (1/Z(x)) exp(-E_θ(r, x) / T)
            
            - T → 0: Deterministic (greedy): r* = argmax_r p_θ(r | x)
            - T = 1: Natural distribution: p_θ(r | x)
            - T > 1: Smoothed distribution (more exploration)
        
        Top-p (nucleus sampling):
            P = {r | Σ_{r'≤r} p_θ(r' | x) ≤ p}
            
            Samples from the smallest set of reasoning paths covering probability mass p.
        
        Self-consistency (N samples):
            p(y | x) = (1/N) Σ_{i=1}^N p_θ(y | x, r_i) where r_i ~ p_θ(r | x)
            
            Variance reduction:
                Var[ŷ] = (1/N) Var[y] → 0 as N → ∞
    
    Quantum annealing schedule:
        T(t) = T₀ · exp(-t/τ)
        
        Where t is the iteration and τ is the annealing time constant.
    
    Attributes:
        num_samples: Number of samples for self-consistency (1 = single trace)
        temperature: Sampling temperature T (0.0 = greedy, >0.0 = sampling)
        top_p: Nucleus sampling parameter p ∈ [0, 1]
        max_reasoning_length: Maximum length of reasoning trace in tokens (T_max)
        max_answer_length: Maximum length of final answer in tokens
        stop_tokens: List of stop tokens/sequences (stopping condition)
        return_reasoning: Whether to return reasoning trace to caller
        decoding_strategy: Decoding strategy (greedy, sampling, nucleus)
        use_self_consistency: Whether to use self-consistency aggregation
        few_shot_examples: Optional few-shot examples for prompt (few-shot learning)
        reasoning_prefix: Prefix to add before reasoning (e.g., "Let's think step by step.")
        answer_prefix: Prefix for final answer (e.g., "Final answer:")
    """
    
    num_samples: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    max_reasoning_length: int = 1000
    max_answer_length: int = 500
    stop_tokens: List[str] = field(default_factory=lambda: ["Final answer:", "Answer:"])
    return_reasoning: bool = True
    decoding_strategy: DecodingStrategy = DecodingStrategy.SAMPLING
    use_self_consistency: bool = False
    few_shot_examples: Optional[List[Dict[str, str]]] = None
    reasoning_prefix: str = "Let's think step by step."
    answer_prefix: str = "Final answer:"


class LLMBackend:
    """
    Abstract interface for LLM backend.
    
    This defines the contract that any LLM implementation must follow
    to work with the CoT framework.
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
    Builds prompts for Chain-of-Thought reasoning.
    
    Mathematical formulation:
        Prompt construction:
            P(x, E) = [S; E₁; ...; E_k; x]
        
        Where:
        - S = system prompt (prior knowledge)
        - E = {E₁, ..., E_k} = few-shot examples
        - x = input question
        
        Conditional probability:
            p_θ(r | P(x, E)) = p_θ(r | x, E)
        
        In-context learning:
            p_θ(y | x, E) = ∫ p_θ(y | x, r) p_θ(r | x, E) dr
        
        Information-theoretic view:
            I(Y; E | X) = H(Y | X) - H(Y | X, E)
            
            Measures how much few-shot examples reduce uncertainty.
    
    Quantum circuit analogy:
        |P⟩ = U_examples · U_system · |x⟩
        
        Where U_examples and U_system are unitary operators encoding examples and system prompt.
    
    Assembles system prompt, task, and few-shot CoT examples.
    Encodes "reasoning mode" (step-by-step, scratchpad format, etc.).
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that reasons step-by-step.
Break down complex problems into smaller steps, show your reasoning process,
and then provide a clear final answer."""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        reasoning_prefix: str = "Let's think step by step.",
        answer_prefix: str = "Final answer:",
    ):
        """
        Initialize the PromptBuilder.
        
        Args:
            system_prompt: Custom system prompt (uses default if None)
            few_shot_examples: List of few-shot examples with 'question' and 'answer' keys
            reasoning_prefix: Prefix to add before reasoning
            answer_prefix: Prefix for final answer
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.few_shot_examples = few_shot_examples or []
        self.reasoning_prefix = reasoning_prefix
        self.answer_prefix = answer_prefix
    
    def build(self, question: str) -> str:
        """
        Build a complete prompt for CoT reasoning.
        
        Args:
            question: The question to answer
            
        Returns:
            Complete prompt string
        """
        parts = [self.system_prompt]
        
        # Add few-shot examples if provided
        if self.few_shot_examples:
            parts.append("\n\nExamples:")
            for example in self.few_shot_examples:
                ex_q = example.get("question", "")
                ex_a = example.get("answer", "")
                parts.append(f"\nQ: {ex_q}")
                parts.append(f"A: {self.reasoning_prefix}\n{ex_a}")
        
        # Add the actual question
        parts.append(f"\n\nQ: {question}")
        parts.append(f"A: {self.reasoning_prefix}")
        
        return "\n".join(parts)


class TraceGenerator:
    """
    Generates reasoning traces by calling the LLM.
    
    Mathematical formulation:
        Generation process:
            r_t ~ p_θ(r_t | r_{1:t-1}, x) for t = 1, ..., T
        
        Decoding strategies:
            1. Greedy (T → 0):
                r_t = argmax_{r_t} p_θ(r_t | r_{1:t-1}, x)
            
            2. Sampling (Boltzmann):
                r_t ~ p_θ(r_t | r_{1:t-1}, x) = softmax(logits / T)
            
            3. Nucleus (top-p):
                r_t ~ p_θ(r_t | r_{1:t-1}, x) · 𝟙[r_t ∈ P_t]
                
                Where P_t = smallest set s.t. Σ_{r'∈P_t} p_θ(r' | r_{1:t-1}, x) ≥ p
        
        Stopping criterion:
            Stop when: r_t ∈ {stop_tokens} or t ≥ T_max
        
        Quantum measurement:
            |r_t⟩ = M_t |ψ_{t-1}⟩
            
            Where M_t is a measurement operator and |ψ_{t-1}⟩ is the superposition state.
    
    Information-theoretic stopping:
        Stop when: I(r_t; Y | r_{1:t-1}, X) < ε
        
        Where ε is a threshold for information gain.
    
    Implements decoding policies: greedy, top-p, temperature, etc.
    Enforces stopping conditions (e.g., special delimiter).
    """
    
    def __init__(
        self,
        llm: LLMBackend,
        config: CoTConfig,
    ):
        """
        Initialize the TraceGenerator.
        
        Args:
            llm: LLM backend instance
            config: CoT configuration
        """
        self.llm = llm
        self.config = config
    
    def generate(
        self,
        prompt: str,
    ) -> CoTTrace:
        """
        Generate a single reasoning trace.
        
        Args:
            prompt: Input prompt
            
        Returns:
            CoTTrace object with reasoning steps and raw text
        """
        # Determine decoding parameters based on strategy
        if self.config.decoding_strategy == DecodingStrategy.GREEDY:
            temperature = 0.0
            top_p = 1.0
        elif self.config.decoding_strategy == DecodingStrategy.NUCLEUS:
            temperature = self.config.temperature
            top_p = self.config.top_p
        elif self.config.decoding_strategy == DecodingStrategy.QUANTUM:
            # Quantum sampling uses temperature for amplitude calculation
            temperature = self.config.temperature
            top_p = 1.0
        else:  # SAMPLING
            temperature = self.config.temperature
            top_p = 1.0
        
        # Generate text from LLM
        max_tokens = (
            self.config.max_reasoning_length + self.config.max_answer_length
        )
        
        try:
            raw_text = self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=self.config.stop_tokens,
            )
        except Exception as e:
            logger.error(f"Error generating trace: {e}")
            raw_text = ""
        
        # Parse reasoning steps from raw text
        steps = self._parse_steps(raw_text)
        
        return CoTTrace(
            steps=steps,
            raw_text=raw_text,
            logprob=0.0,  # Would need model logprobs to compute
        )
    
    def _parse_steps(self, text: str) -> List[CoTStep]:
        """
        Parse reasoning steps from raw text.
        
        Attempts to extract structured steps from the reasoning text.
        Falls back to splitting by common patterns if structured format not found.
        
        Args:
            text: Raw reasoning text
            
        Returns:
            List of CoTStep objects
        """
        steps = []
        
        # Try to find structured step patterns
        # Pattern 1: "Step 1:", "Step 2:", etc.
        step_pattern = r"(?:Step\s+\d+[:.]|Step\s+\d+[:.]|^\d+[.)]\s+)(.+?)(?=(?:Step\s+\d+[:.]|Step\s+\d+[:.]|^\d+[.)]\s+|Final answer:|Answer:|$))"
        matches = re.finditer(step_pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        if matches:
            for idx, match in enumerate(matches, start=1):
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(CoTStep(index=idx, text=step_text))
        
        # Pattern 2: "Thought:", "Reasoning:", etc.
        if not steps:
            thought_pattern = r"(?:Thought|Reasoning|Analysis)[:\s]+(.+?)(?=(?:Thought|Reasoning|Analysis|Final answer|Answer)[:\s]|$)"
            matches = re.finditer(thought_pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            
            for idx, match in enumerate(matches, start=1):
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(CoTStep(index=idx, text=step_text))
        
        # Fallback: split by sentences or newlines if no structured format
        if not steps:
            # Remove answer prefix if present
            reasoning_text = text
            for prefix in ["Final answer:", "Answer:"]:
                if prefix.lower() in reasoning_text.lower():
                    reasoning_text = reasoning_text.split(prefix, 1)[0]
            
            # Split by double newlines or periods followed by space
            sentences = re.split(r'(?:\n\n|\.\s+(?=[A-Z]))', reasoning_text)
            for idx, sentence in enumerate(sentences, start=1):
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Filter very short fragments
                    steps.append(CoTStep(index=idx, text=sentence))
        
        # If still no steps, create one from entire text
        if not steps:
            steps.append(CoTStep(index=1, text=text.strip()))
        
        return steps


class AnswerDecoder:
    """
    Decodes the final answer from the reasoning trace.
    
    Mathematical formulation:
        Answer extraction:
            y* = argmax_{y} p_θ(y | x, r)
        
        Marginalization over reasoning:
            p(y | x) = Σ_{r} p_θ(r | x) · p_θ(y | x, r)
        
        Maximum a posteriori (MAP):
            y* = argmax_{y} p(y | x) = argmax_{y} Σ_{r} p_θ(r | x) · p_θ(y | x, r)
    
    Quantum measurement:
        P(y | x) = |⟨y | ψ⟩|² = |Σ_{r: y_r=y} α_r|²
        
        Where |ψ⟩ = Σ_{r} α_r |r⟩ ⊗ |y_r⟩ is the entangled state.
    
    Information extraction:
        y* = argmax_{y} I(Y; R | X)
        
        Maximizes mutual information between answer and reasoning.
    
    Supports two modes:
    - Option A: Generate reasoning and final answer in a single pass
      y* = decode(r*) where r* ~ p_θ(r | x)
    
    - Option B: Two-pass (first reasoning, then answer)
      Step 1: r* ~ p_θ(r | x)
      Step 2: y* ~ p_θ(y | x, r*)
    """
    
    def __init__(self, answer_prefix: str = "Final answer:"):
        """
        Initialize the AnswerDecoder.
        
        Args:
            answer_prefix: Prefix that indicates the start of the final answer
        """
        self.answer_prefix = answer_prefix
    
    def decode(self, trace: CoTTrace) -> str:
        """
        Extract the final answer from a reasoning trace.
        
        Args:
            trace: CoTTrace object
            
        Returns:
            Final answer string
        """
        raw_text = trace.raw_text
        
        # Try to find answer after prefix
        for prefix in [self.answer_prefix, "Answer:", "Final Answer:"]:
            if prefix.lower() in raw_text.lower():
                # Find the prefix (case-insensitive)
                idx = raw_text.lower().find(prefix.lower())
                if idx != -1:
                    answer = raw_text[idx + len(prefix):].strip()
                    # Remove any trailing reasoning that might have leaked
                    # Stop at common delimiters
                    answer = re.split(r'\n\n|Thought:|Reasoning:', answer)[0].strip()
                    if answer:
                        return answer
        
        # If no explicit answer prefix found, try to extract from last step
        if trace.steps:
            last_step = trace.steps[-1].text
            # Look for patterns like "Therefore, ..." or "So the answer is ..."
            patterns = [
                r"(?:Therefore|So|Thus|Hence|In conclusion)[,:\s]+(.+?)(?:\.|$)",
                r"(?:answer|solution|result)\s+is[:\s]+(.+?)(?:\.|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, last_step, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # Fallback: return last step text or empty string
        if trace.steps:
            return trace.steps[-1].text.strip()
        
        return raw_text.strip()


class TraceEvaluator:
    """
    Optional verifier to score the quality of a reasoning trace.
    
    Mathematical formulation:
        Score function:
            s(r, x, y) ∈ [0, 1]
        
        Energy-based scoring:
            s(r, x, y) = exp(-E_verifier(r, x, y) / T_verifier)
            
            Where E_verifier is the verifier's energy function.
        
        Learned scoring:
            s(r, x, y) = σ(f_φ(r, x, y))
            
            Where f_φ is a neural network and σ is sigmoid.
        
        Consistency check:
            s(r, x, y) = 𝟙[consistent(r, y)] · correctness(r, x, y)
            
            Where consistent(r, y) checks logical consistency.
    
    Information-theoretic scoring:
        s(r, x, y) = I(R; Y | X) / H(Y | X)
        
        Measures how much reasoning reduces answer uncertainty.
    
    Quantum fidelity:
        s(r, x, y) = |⟨y | U_r |x⟩|²
        
        Where U_r is the unitary operator representing reasoning r.
    
    Can use:
    - Regex checks (does arithmetic add up?)
      s = 𝟙[∀(a op b = c) in r: verify(a op b = c)]
    
    - Program execution
      s = 𝟙[execute(r) produces y]
    
    - Second model that judges consistency
      s = p_verifier(consistent | r, x, y)
    """
    
    def __init__(self, evaluator_type: str = "heuristic"):
        """
        Initialize the TraceEvaluator.
        
        Args:
            evaluator_type: Type of evaluator ("heuristic", "regex", "llm")
        """
        self.evaluator_type = evaluator_type
    
    def score(
        self,
        question: str,
        trace: CoTTrace,
    ) -> float:
        """
        Score the quality of a reasoning trace.
        
        Args:
            question: Original question
            trace: Reasoning trace to evaluate
            
        Returns:
            Score between 0.0 and 1.0
        """
        if self.evaluator_type == "heuristic":
            return self._heuristic_score(trace)
        elif self.evaluator_type == "regex":
            return self._regex_score(trace)
        else:
            # Default: basic heuristic
            return self._heuristic_score(trace)
    
    def _heuristic_score(self, trace: CoTTrace) -> float:
        """
        Heuristic scoring based on trace properties.
        
        Uses energy-based scoring: s(r, x, y) = exp(-E_verifier(r, x, y) / T_verifier)
        
        Args:
            trace: Reasoning trace
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Reward having multiple steps
        if len(trace.steps) > 1:
            score += 0.3
        
        # Reward reasonable step length (not too short, not too long)
        avg_step_length = sum(len(s.text) for s in trace.steps) / max(len(trace.steps), 1)
        if 50 <= avg_step_length <= 500:
            score += 0.3
        
        # Reward having structured format
        if any("step" in s.text.lower()[:20] for s in trace.steps):
            score += 0.2
        
        # Reward having a conclusion/answer
        if trace.raw_text.lower().count("answer") > 0 or trace.raw_text.lower().count("therefore") > 0:
            score += 0.2
        
        # Convert to energy-based score if logprob is available
        if trace.logprob != 0.0:
            # Energy: E = -logprob
            energy = EnergyFunction.calculate_energy(trace.logprob)
            # Normalize energy to [0, 1] range (assuming reasonable bounds)
            normalized_energy = min(1.0, max(0.0, energy / 10.0))  # Scale factor
            energy_score = math.exp(-normalized_energy)
            # Combine heuristic and energy-based scores
            score = 0.7 * score + 0.3 * energy_score
        
        return min(score, 1.0)
    
    def _regex_score(self, trace: CoTTrace) -> float:
        """
        Regex-based scoring (e.g., checking arithmetic consistency).
        
        Args:
            trace: Reasoning trace
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Example: check for arithmetic consistency
        # Pattern: "X + Y = Z" or "X * Y = Z"
        arithmetic_pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        
        score = 0.5  # Base score
        
        for step in trace.steps:
            matches = re.finditer(arithmetic_pattern, step.text)
            for match in matches:
                try:
                    a = float(match.group(1))
                    op = match.group(2)
                    b = float(match.group(3))
                    expected = float(match.group(4))
                    
                    if op == '+':
                        result = a + b
                    elif op == '-':
                        result = a - b
                    elif op == '*':
                        result = a * b
                    elif op == '/':
                        result = a / b if b != 0 else float('inf')
                    else:
                        continue
                    
                    # Check if result matches expected (with small tolerance)
                    if abs(result - expected) < 0.01:
                        score += 0.1
                    else:
                        score -= 0.1
                except (ValueError, ZeroDivisionError):
                    continue
        
        return max(0.0, min(1.0, score))


class SelfConsistencyEngine:
    """
    Self-consistency engine for aggregating multiple reasoning traces.
    
    Mathematical formulation:
        Ensemble aggregation:
            p(y | x) = (1/N) Σ_{i=1}^N p_θ(y | x, r_i)
            
            Where r_i ~ p_θ(r | x) are independent samples.
        
        Weighted aggregation:
            p(y | x) = (1/Z) Σ_{i=1}^N w_i · p_θ(y | x, r_i)
            
            Where:
            - w_i = score(r_i) or p_θ(r_i | x)
            - Z = Σ_{i=1}^N w_i (normalization)
        
        Majority voting:
            ŷ = argmax_{y} Σ_{i=1}^N 𝟙[y_i = y]
        
        Weighted voting:
            ŷ = argmax_{y} Σ_{i=1}^N w_i · 𝟙[y_i = y]
        
        Confidence estimation:
            Confidence = max_{y} p(y | x) = max_{y} (1/N) Σ_{i=1}^N 𝟙[y_i = y]
            
            Or using entropy:
                Confidence = 1 - H(Y | X) / log |Y|
                
                Where H(Y | X) = -Σ_{y} p(y | x) log p(y | x)
    
    Quantum ensemble:
        |ψ_ensemble⟩ = (1/√N) Σ_{i=1}^N |r_i⟩ ⊗ |y_i⟩
        
        Measurement:
            P(y | x) = |⟨y | ψ_ensemble⟩|² = (1/N) |Σ_{i: y_i=y} 1|²
    
    Variance reduction:
        Var[ŷ] = (1/N) Var[y] → 0 as N → ∞
        
        By Central Limit Theorem:
            ŷ → E[y] as N → ∞ (almost surely)
    
    Samples multiple traces and aggregates answers using majority voting
    or weighted voting based on trace quality scores.
    """
    
    def __init__(
        self,
        use_verifier: bool = False,
        verifier: Optional[TraceEvaluator] = None,
    ):
        """
        Initialize the SelfConsistencyEngine.
        
        Args:
            use_verifier: Whether to use verifier scores for weighted voting
            verifier: Optional TraceEvaluator instance
        """
        self.use_verifier = use_verifier
        self.verifier = verifier
    
    def aggregate(
        self,
        question: str,
        traces: List[CoTTrace],
    ) -> Tuple[str, float]:
        """
        Aggregate multiple reasoning traces into a final answer.
        
        Implements self-consistency: marginalize over reasoning paths
        and pick the most common answer.
        
        Args:
            question: Original question
            traces: List of reasoning traces
            
        Returns:
            Tuple of (final_answer, confidence)
        """
        if not traces:
            return "", 0.0
        
        # Extract answers from each trace
        answers = []
        weights = []
        
        decoder = AnswerDecoder()
        
        for trace in traces:
            answer = decoder.decode(trace)
            # Normalize answer (lowercase, strip whitespace)
            normalized = answer.lower().strip()
            
            if normalized:
                answers.append(normalized)
                
                # Compute weight (use verifier score if available)
                if self.use_verifier and self.verifier:
                    weight = self.verifier.score(question, trace)
                    weights.append(weight)
                else:
                    weights.append(1.0)
        
        if not answers:
            return "", 0.0
        
        # Count answers (with or without weights)
        if self.use_verifier and any(w > 0 for w in weights):
            # Weighted voting: ŷ = argmax_y Σ_{i=1}^N w_i · 𝟙[y_i = y]
            answer_counts: Dict[str, float] = {}
            for answer, weight in zip(answers, weights):
                answer_counts[answer] = answer_counts.get(answer, 0.0) + weight
            
            # Get answer with highest weighted count
            final_answer_normalized = max(answer_counts.items(), key=lambda x: x[1])[0]
            total_weight = sum(answer_counts.values())
            confidence = answer_counts[final_answer_normalized] / total_weight if total_weight > 0 else 0.0
            
            # Calculate confidence using entropy: C = 1 - H(Y | X) / log |Y|
            answer_probs = {ans: count / total_weight for ans, count in answer_counts.items()}
            if len(answer_probs) > 1:
                entropy = InformationTheory.entropy(list(answer_probs.values()))
                max_entropy = math.log2(len(answer_probs))
                if max_entropy > 0:
                    entropy_confidence = 1.0 - (entropy / max_entropy)
                    # Combine weighted and entropy-based confidence
                    confidence = 0.7 * confidence + 0.3 * entropy_confidence
        else:
            # Simple majority voting: ŷ = argmax_y Σ_{i=1}^N 𝟙[y_i = y]
            answer_counts = Counter(answers)
            final_answer_normalized, count = answer_counts.most_common(1)[0]
            confidence = count / len(answers)
            
            # Calculate confidence using entropy
            if len(answer_counts) > 1:
                answer_probs = [count / len(answers) for count in answer_counts.values()]
                entropy = InformationTheory.entropy(answer_probs)
                max_entropy = math.log2(len(answer_counts))
                if max_entropy > 0:
                    entropy_confidence = 1.0 - (entropy / max_entropy)
                    confidence = 0.7 * confidence + 0.3 * entropy_confidence
        
        # Find original answer (preserving case) for the normalized answer
        # Use the first occurrence that matches
        for trace in traces:
            answer = decoder.decode(trace)
            if answer.lower().strip() == final_answer_normalized:
                return answer, confidence
        
        # Fallback: return normalized answer (will be lowercase)
        return final_answer_normalized, confidence


class CoTReasoner:
    """
    Main Chain-of-Thought reasoning engine.
    
    Mathematical pipeline:
        Step 1: Prompt construction
            P = PromptBuilder(x, E) where E = few-shot examples
        
        Step 2: Reasoning generation
            r* ~ p_θ(r | P) = p_θ(r | x, E)
            
            Or multiple samples: {r₁, ..., r_N} ~ p_θ(r | P)
        
        Step 3: Answer extraction
            y* = AnswerDecoder.decode(r*)
            
            Or marginalized: y* = argmax_y Σ_{i=1}^N p_θ(y | x, r_i)
        
        Step 4: Self-consistency (if N > 1)
            ŷ = SelfConsistencyEngine.aggregate({r₁, ..., r_N})
            Confidence = max_y p(y | x)
    
    Complete probabilistic model:
        p_θ(y | x) = Σ_{r} p_θ(r | x) · p_θ(y | x, r)
        
        With self-consistency (Monte Carlo estimate):
            p_θ(y | x) ≈ (1/N) Σ_{i=1}^N p_θ(y | x, r_i) where r_i ~ p_θ(r | x)
    
    Variational inference view:
        ELBO: log p_θ(y | x) ≥ E_{q_φ(r|x,y)}[log p_θ(y | x, r)] - KL(q_φ || p_θ)
        
        Where q_φ is approximated by the sampling distribution.
    
    Quantum circuit:
        |x⟩ → U_prompt → |P⟩ → U_reasoning → |r⟩ → U_decode → |y⟩
        
        Full evolution:
            |y⟩ = U_decode · U_reasoning · U_prompt |x⟩
    
    Implements the core CoT algorithm:
    1. Build prompt with few-shot examples
    2. Generate reasoning trace(s)
    3. Extract final answer
    4. Optionally aggregate multiple traces (self-consistency)
    """
    
    def __init__(
        self,
        llm: LLMBackend,
        config: Optional[CoTConfig] = None,
        verifier: Optional[TraceEvaluator] = None,
    ):
        """
        Initialize the CoTReasoner.
        
        Args:
            llm: LLM backend instance
            config: CoT configuration (uses defaults if None)
            verifier: Optional trace evaluator
        """
        self.llm = llm
        self.config = config or CoTConfig()
        self.verifier = verifier
        
        # Initialize components
        self.prompt_builder = PromptBuilder(
            few_shot_examples=self.config.few_shot_examples,
            reasoning_prefix=self.config.reasoning_prefix,
            answer_prefix=self.config.answer_prefix,
        )
        self.trace_generator = TraceGenerator(llm=llm, config=self.config)
        self.answer_decoder = AnswerDecoder(
            answer_prefix=self.config.answer_prefix
        )
        
        # Initialize self-consistency engine if needed
        use_verifier = verifier is not None
        self.consistency_engine = SelfConsistencyEngine(
            use_verifier=use_verifier,
            verifier=verifier,
        )
    
    def solve(
        self,
        question: Union[str, Question],
    ) -> CoTResult:
        """
        Solve a question using Chain-of-Thought reasoning.
        
        Args:
            question: Question string or Question object
            
        Returns:
            CoTResult with reasoning traces and final answer
        """
        # Convert to Question object if needed
        if isinstance(question, str):
            question_obj = Question(id="", text=question)
        else:
            question_obj = question
        
        # Build prompt
        prompt = self.prompt_builder.build(question_obj.text)
        
        # Generate traces
        num_samples = self.config.num_samples
        if self.config.use_self_consistency and num_samples == 1:
            num_samples = 3  # Default to 3 for self-consistency
        
        traces = []
        for _ in range(num_samples):
            trace = self.trace_generator.generate(prompt)
            
            # Optionally score the trace
            if self.verifier:
                trace.score = self.verifier.score(question_obj.text, trace)
            
            traces.append(trace)
        
        # Apply quantum sampling if configured
        if self.config.decoding_strategy == DecodingStrategy.QUANTUM and len(traces) > 1:
            # Calculate probabilities from logprobs or scores
            if all(t.logprob != 0.0 for t in traces):
                probs = [math.exp(t.logprob) for t in traces]
            elif all(t.score is not None for t in traces):
                probs = [t.score if t.score is not None else 0.0 for t in traces]
            else:
                probs = [1.0 / len(traces)] * len(traces)
            
            # Quantum sampling
            traces = QuantumSampler.quantum_sampling(traces, probs, num_samples)
        
        # Apply Boltzmann sampling if temperature > 0 and multiple traces
        elif self.config.temperature > 0 and len(traces) > 1:
            # Use Boltzmann sampling for re-weighting
            sampled_traces = EnergyFunction.boltzmann_sampling(
                traces, self.config.temperature, num_samples
            )
            if sampled_traces:
                traces = sampled_traces
        
        # Extract final answer
        if num_samples == 1:
            # Single trace: just decode
            final_answer = self.answer_decoder.decode(traces[0])
            confidence = traces[0].score if traces[0].score is not None else 0.5
        else:
            # Multiple traces: aggregate using self-consistency
            # Use quantum measurement if quantum decoding strategy
            if self.config.decoding_strategy == DecodingStrategy.QUANTUM:
                answers = [self.answer_decoder.decode(t) for t in traces]
                # Calculate probabilities from logprobs or scores
                if all(t.logprob != 0.0 for t in traces):
                    probs = [math.exp(t.logprob) for t in traces]
                elif all(t.score is not None for t in traces):
                    probs = [t.score if t.score is not None else 0.0 for t in traces]
                else:
                    probs = [1.0 / len(traces)] * len(traces)
                
                # Quantum measurement: P(y | x) = |⟨y | ψ⟩|²
                final_answer, confidence = QuantumSampler.measure_state(traces, answers, probs)
            else:
                # Standard self-consistency aggregation
                final_answer, confidence = self.consistency_engine.aggregate(
                    question_obj.text, traces
                )
        
        # Calculate additional metrics using mathematical utilities
        extra_metrics = {
            "num_traces": len(traces),
            "avg_trace_length": sum(len(t.raw_text) for t in traces) / max(len(traces), 1),
        }
        
        # Add information-theoretic metrics
        if len(traces) > 1:
            trace_entropy = InformationTheory.calculate_trace_entropy(traces)
            extra_metrics["trace_entropy"] = trace_entropy
            
            # Calculate partition function and free energy if logprobs available
            if all(t.logprob != 0.0 for t in traces):
                energies = [EnergyFunction.calculate_energy(t.logprob) for t in traces]
                partition_func = EnergyFunction.partition_function(energies, self.config.temperature)
                free_energy = EnergyFunction.free_energy(partition_func, self.config.temperature)
                extra_metrics["partition_function"] = partition_func
                extra_metrics["free_energy"] = free_energy
        
        # Find shortest path if multiple traces
        if len(traces) > 1:
            shortest_path = GraphReasoning.find_shortest_path(traces, lambda_reg=0.1)
            if shortest_path:
                extra_metrics["shortest_path_length"] = len(shortest_path.steps)
        
        return CoTResult(
            question=question_obj,
            traces=traces if self.config.return_reasoning else [],
            final_answer=final_answer,
            confidence=confidence,
            extra_metrics=extra_metrics,
        )


class CoTAgent:
    """
    Chain-of-Thought Agent for step-by-step reasoning.
    
    This agent implements the Chain-of-Thought (CoT) reasoning framework,
    which introduces an explicit latent sequence of reasoning tokens between
    input and output, and searches over that latent space with a sequence model.
    
    Mathematical Foundation:
        Core model:
            p_θ(y, r | x) = p_θ(r | x) · p_θ(y | x, r)
        
        Where:
        - x = input (question, task description) ∈ X
        - y = final answer ∈ Y
        - r = (r₁, ..., r_T) = reasoning trace (CoT), a sequence of tokens
        - θ = model parameters
    
        Variational lower bound:
            log p_θ(y | x) ≥ E_{q_φ(r|x,y)}[log p_θ(y | x, r)] - KL(q_φ(r|x,y) || p_θ(r|x))
    
        Information-theoretic:
            I(X; Y | R) = H(Y | R) - H(Y | X, R)
            
            Mutual information between input and output given reasoning.
    
        Quantum superposition:
            |ψ⟩ = Σ_{r} α_r |r⟩ ⊗ |y_r⟩ where α_r = √(p_θ(r | x))
            
            Measurement: P(y | x) = |⟨y | ψ⟩|²
    
        Statistical mechanics:
            p_θ(r | x) = (1/Z(x)) exp(-E_θ(r, x) / T)
            
            Where E_θ(r, x) = -log p_θ(r | x) is the energy function.
    
        Self-consistency:
            p(y | x) = (1/N) Σ_{i=1}^N p_θ(y | x, r_i) where r_i ~ p_θ(r | x)
            
            Confidence: C = 1 - H(Y | X) / log |Y|
    
    Computational complexity:
        Time: O(N · T · |V| · d) where N = samples, T = trace length, |V| = vocab, d = dimension
        Space: O(N · T · d) for storing traces
    
    Attributes:
        agent_name: Name of the agent
        model_name: LLM model to use
        config: CoT configuration (temperature T, top_p, etc.)
        verifier: Optional trace evaluator (energy-based or learned)
        reasoner: Internal CoTReasoner instance
    
    Example:
        >>> from swarms.agents import CoTAgent
        >>> agent = CoTAgent(
        ...     agent_name="cot-agent",
        ...     model_name="gpt-4o",
        ... )
        >>> result = agent.run("Solve step by step: What is 15 * 23?")
        >>> print(result)
    """
    
    def __init__(
        self,
        agent_name: str = "cot-agent",
        model_name: str = "gpt-4o",
        system_prompt: Optional[str] = None,
        config: Optional[CoTConfig] = None,
        verifier: Optional[TraceEvaluator] = None,
        agent: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the CoTAgent.
        
        Args:
            agent_name: Name of the agent
            model_name: LLM model name (used if agent not provided)
            system_prompt: Optional custom system prompt
            config: CoT configuration (uses defaults if None)
            verifier: Optional trace evaluator
            agent: Optional Agent instance to use (if provided, uses its LLM)
            **kwargs: Additional arguments passed to Agent if creating one
        """
        self.agent_name = agent_name
        self.model_name = model_name
        self.config = config or CoTConfig()
        self.verifier = verifier
        
        # If agent is provided, use it; otherwise create adapter from model
        if agent is not None:
            self.agent = agent
            llm_adapter = AgentLLMAdapter(agent)
        else:
            # Import Agent here to avoid circular imports
            from swarms.structs.agent import Agent
            
            self.agent = Agent(
                agent_name=agent_name,
                model_name=model_name,
                system_prompt=system_prompt,
                **kwargs,
            )
            llm_adapter = AgentLLMAdapter(self.agent)
        
        # Initialize the CoT reasoner
        self.reasoner = CoTReasoner(
            llm=llm_adapter,
            config=self.config,
            verifier=self.verifier,
        )
    
    def run(
        self,
        task: str,
        return_reasoning: Optional[bool] = None,
    ) -> Union[str, CoTResult]:
        """
        Run the Chain-of-Thought agent on a task.
        
        Args:
            task: Task or question to solve
            return_reasoning: Whether to return full CoTResult (defaults to config setting)
            
        Returns:
            Final answer string, or CoTResult if return_reasoning=True
        """
        # Temporarily override return_reasoning if specified
        original_return_reasoning = self.config.return_reasoning
        if return_reasoning is not None:
            self.config.return_reasoning = return_reasoning
        
        try:
            result = self.reasoner.solve(task)
            
            # Return based on configuration
            if self.config.return_reasoning:
                return result
            else:
                return result.final_answer
        finally:
            # Restore original setting
            self.config.return_reasoning = original_return_reasoning


class AgentLLMAdapter(LLMBackend):
    """
    Adapter to use Agent's LLM with the CoT framework.
    
    Wraps the Agent's LLM interface to match the LLMBackend contract.
    """
    
    def __init__(self, agent: Any):
        """
        Initialize the adapter.
        
        Args:
            agent: Agent instance with an LLM
        """
        self.agent = agent
        self.llm = agent.llm if hasattr(agent, 'llm') else None
    
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
        if self.llm is None:
            raise ValueError("Agent does not have an LLM configured")
        
        try:
            # Try to use the LLM's run method directly
            if hasattr(self.llm, 'run'):
                # Store original temperature/top_p if they exist
                original_temp = getattr(self.llm, 'temperature', None)
                original_top_p = getattr(self.llm, 'top_p', None)
                original_max_tokens = getattr(self.llm, 'max_tokens', None)
                
                # Temporarily set parameters
                if hasattr(self.llm, 'temperature'):
                    self.llm.temperature = temperature
                if hasattr(self.llm, 'top_p'):
                    self.llm.top_p = top_p
                if hasattr(self.llm, 'max_tokens'):
                    self.llm.max_tokens = max_tokens
                
                try:
                    result = self.llm.run(prompt, stop=stop)
                finally:
                    # Restore original parameters
                    if original_temp is not None and hasattr(self.llm, 'temperature'):
                        self.llm.temperature = original_temp
                    if original_top_p is not None and hasattr(self.llm, 'top_p'):
                        self.llm.top_p = original_top_p
                    if original_max_tokens is not None and hasattr(self.llm, 'max_tokens'):
                        self.llm.max_tokens = original_max_tokens
                
                return result if isinstance(result, str) else str(result)
            
            # Fallback: try calling the LLM directly
            elif callable(self.llm):
                return str(self.llm(prompt))
            
            # Last resort: use agent's run method
            else:
                return str(self.agent.run(prompt))
        
        except Exception as e:
            logger.error(f"Error in AgentLLMAdapter.generate: {e}")
            # Fallback to agent's run method
            try:
                return str(self.agent.run(prompt))
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return ""


def apply_cot_to_agent(
    agent: Any,
    task: str,
    cot_config: Optional[CoTConfig] = None,
) -> str:
    """
    Apply Chain-of-Thought reasoning to an Agent's task.
    
    This function integrates CoT into the Agent workflow when chain_of_thoughts=True.
    It wraps the Agent's LLM, runs CoT reasoning, and returns the final answer.
    
    Args:
        agent: Agent instance
        task: Task/question to solve
        cot_config: Optional CoT configuration (uses defaults if None)
        
    Returns:
        Final answer string
    """
    # Create CoTAgent with the provided agent
    cot_agent = CoTAgent(
        agent=agent,
        config=cot_config,
    )
    
    # Run and return just the answer
    return cot_agent.run(task, return_reasoning=False)


# Main exports - only export the essential class
# All other classes are internal implementation details
__all__ = [
    "CoTAgent",
]

