from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from uuid import uuid4, UUID
import math
import random
from collections import deque, Counter

from loguru import logger


class SearchStrategy(str, Enum):
    BEAM = "beam"
    BFS = "bfs"
    DFS = "dfs"
    MCTS = "mcts"
    QUANTUM = "quantum"  # Quantum-inspired tree search


class TreeInformationTheory:
    @staticmethod
    def path_entropy(path_probs: List[float]) -> float:
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
        return prior_entropy - conditional_entropy
    
    @staticmethod
    def expected_information_gain(
        node_probs: List[float],
        child_entropies: List[float]
    ) -> float:
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
    @staticmethod
    def calculate_path_amplitudes(path_probs: List[float]) -> List[float]:
        return [math.sqrt(max(0.0, p)) for p in path_probs]
    
    @staticmethod
    def quantum_measurement(
        paths: List[List[str]],
        answers: List[str],
        path_probs: Optional[List[float]] = None
    ) -> Tuple[str, float]:
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
    @staticmethod
    def calculate_path_energy(path_logprob: float) -> float:
        return -path_logprob
    
    @staticmethod
    def boltzmann_path_weight(energy: float, temperature: float) -> float:
        if temperature <= 0:
            return 0.0 if energy > 0 else 1.0
        
        return math.exp(-energy / temperature)
    
    @staticmethod
    def tree_partition_function(
        path_energies: List[float],
        temperature: float
    ) -> float:
        if temperature <= 0:
            return 1.0
        
        weights = [TreeEnergyFunction.boltzmann_path_weight(e, temperature) for e in path_energies]
        return sum(weights)
    
    @staticmethod
    def tree_free_energy(partition_function: float, temperature: float) -> float:
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
    @staticmethod
    def calculate_path_probability(
        node_scores: List[float],
        normalize: bool = True
    ) -> float:
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
    @staticmethod
    def calculate_ucb1(
        node_value: float,
        node_visits: int,
        parent_visits: int,
        exploration_constant: float = math.sqrt(2)
    ) -> float:
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
        new_visits = current_visits + 1
        updated_value = (current_value * current_visits + new_value) / new_visits
        
        return updated_value, new_visits


@dataclass
class ThoughtNode:
    
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
        if self.parent is None:
            return self.text
        return self.parent.get_path() + self.text
    
    def get_average_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class ToTConfig:
    
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
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        raise NotImplementedError("Subclass must implement generate method")


class PromptBuilder:
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that reasons through problems
by exploring multiple possible approaches. Consider different perspectives and evaluate
which reasoning paths are most promising."""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
    ):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
    
    def build_initial_prompt(self, problem: str) -> str:
        return f"{self.system_prompt}\n\nProblem: {problem}\n\nLet's explore different approaches to solve this problem."
    
    def build_expansion_prompt(
        self,
        problem: str,
        partial_reasoning: str,
        num_branches: int,
    ) -> str:
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
        return f"""{self.system_prompt}

Problem: {problem}

Reasoning:
{reasoning}

Based on the reasoning above, provide the final answer.
Format: {self.answer_prefix} [your answer]"""
    
    @property
    def answer_prefix(self) -> str:
        return "Final answer:"


class NodeExpander:
    def __init__(
        self,
        llm: LLMBackend,
        config: ToTConfig,
        prompt_builder: PromptBuilder,
    ):
        self.llm = llm
        self.config = config
        self.prompt_builder = prompt_builder
    
    def expand(
        self,
        problem: str,
        node: ThoughtNode,
    ) -> List[str]:
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
    def __init__(
        self,
        llm: LLMBackend,
        config: ToTConfig,
        prompt_builder: PromptBuilder,
    ):
        self.llm = llm
        self.config = config
        self.prompt_builder = prompt_builder
    
    def evaluate(
        self,
        problem: str,
        node: ThoughtNode,
    ) -> float:
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
    def __init__(
        self,
        llm: LLMBackend,
        prompt_builder: PromptBuilder,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
    
    def extract(
        self,
        problem: str,
        reasoning: str,
    ) -> str:
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
    def __init__(
        self,
        config: ToTConfig,
        expander: NodeExpander,
        evaluator: Evaluator,
    ):
        self.config = config
        self.expander = expander
        self.evaluator = evaluator
    
    def search(
        self,
        problem: str,
        root: ThoughtNode,
    ) -> List[ThoughtNode]:
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
    def __init__(
        self,
        llm: LLMBackend,
        config: Optional[ToTConfig] = None,
    ):
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
        description: Optional[str] = None,
        model_name: Optional[str] = "gpt-4o",
        llm: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        global_system_prompt: Optional[str] = None,
        secondary_system_prompt: Optional[str] = None,
        config: Optional[ToTConfig] = None,
        agent: Optional[Any] = None,
        **kwargs,
    ):
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
            llm_adapter = AgentLLMAdapter(self.agent)
        
        # Initialize the ToT reasoner
        self.reasoner = ToTReasoner(
            llm=llm_adapter,
            config=self.config,
        )
    
    def step(self, task: str, *args, **kwargs) -> str:
        result = self.run(task, return_tree=False, *args, **kwargs)
        return result if isinstance(result, str) else result.get("answer", "")
    
    def __getattr__(self, name: str):
        if hasattr(self, 'agent'):
            return getattr(self.agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def run(
        self,
        task: str,
        return_tree: Optional[bool] = None,
    ) -> Union[str, Dict[str, Any]]:
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

