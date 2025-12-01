"""
MAKER: Massively decomposed Agentic processes with first-to-ahead-by-K Error correction and Red-flagging

This module implements the MAKER framework from the paper:
"Solving a Million-Step LLM Task with Zero Errors" by Meyerson et al. (2025)

MAKER is a general-purpose framework for solving long-horizon tasks with extreme precision through:
1. MAD (Maximal Agentic Decomposition): Breaking tasks into minimal subtasks
2. First-to-ahead-by-K Voting: Error correction through voting
3. Red-flagging: Discarding unreliable responses

The framework enables solving tasks with millions of LLM steps with zero errors
by exploiting the modularity of extreme decomposition to apply error correction
at each step.

Paper: https://arxiv.org/abs/2511.09030
"""

import uuid
import math
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="maker")


class MAKER:
    """
    MAKER: Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging.

    A general-purpose framework for solving long-horizon tasks with extreme precision
    through massive decomposition of tasks into subtasks, each solved by focused
    microagents with error correction through voting.

    This implementation follows the MAKER framework from the paper:
    "Solving a Million-Step LLM Task with Zero Errors" by Meyerson et al. (2025)

    The framework consists of three core components:

    1. MAD (Maximal Agentic Decomposition):
       By breaking a task with s steps into s subtasks, each agent can focus on a
       single step, reducing context confusion and improving reliability.

    2. First-to-ahead-by-K Voting:
       For each step, multiple samples are drawn until one candidate action is
       K votes ahead of all others, ensuring high probability of correctness.

    3. Red-flagging:
       Responses that show signs of unreliability (overly long or incorrectly
       formatted) are discarded, reducing correlated errors.

    The framework is task-agnostic. Users provide:
    - A task/objective to complete (main input to run())
    - A function to format prompts for each step
    - A function to parse responses and extract the action/result
    - A function to validate responses (for red-flagging)
    - Optional: A function to update state between steps

    Attributes:
        id (str): Unique identifier for the MAKER instance.
        name (str): Human-readable name for the system.
        description (str): Description of the system's purpose.
        model_name (str): Name of the LLM model to use.
        k (int): Vote threshold - candidate must be k votes ahead to win.
        max_tokens (int): Maximum tokens for LLM response (red-flag threshold).
        temperature (float): Temperature for LLM sampling.
        temperature_first (float): Temperature for first vote (typically 0 for determinism).
        system_prompt (str): System prompt for the microagents.
        format_prompt (Callable): Function to format the prompt for each step.
        parse_response (Callable): Function to parse LLM response into a result.
        validate_response (Callable): Function to validate response format (red-flagging).
        update_state (Callable): Function to update state after each step.
        max_workers (int): Maximum parallel workers for concurrent sampling.
        verbose (bool): Whether to enable verbose logging.

    Example:
        >>> from swarms.structs.maker import MAKER
        >>>
        >>> # Define task-specific functions
        >>> def format_prompt(task, state, step_idx, previous_result):
        ...     return f"Task: {task}\\nState: {state}\\nStep {step_idx+1}: What's next?"
        >>>
        >>> def parse_response(response):
        ...     return response.strip()
        >>>
        >>> def validate_response(response, max_tokens):
        ...     return len(response) < max_tokens * 4 and response.strip() != ""
        >>>
        >>> # Create MAKER instance
        >>> maker = MAKER(
        ...     name="MyTaskSolver",
        ...     model_name="gpt-4o-mini",
        ...     system_prompt="You solve tasks step by step.",
        ...     format_prompt=format_prompt,
        ...     parse_response=parse_response,
        ...     validate_response=validate_response,
        ...     k=3,
        ... )
        >>>
        >>> # Run the solver with your task
        >>> results = maker.run(
        ...     task="Calculate the factorial of 5 step by step",
        ...     max_steps=5
        ... )

    References:
        Meyerson, E., et al. (2025). Solving a Million-Step LLM Task with Zero Errors.
        arXiv:2511.09030
    """

    def __init__(
        self,
        id: str = None,
        name: str = "MAKER",
        description: str = "Massively decomposed Agentic processes with Error correction and Red-flagging",
        model_name: str = "gpt-4o-mini",
        system_prompt: str = "You are a precise assistant that solves tasks step by step. Follow instructions exactly and provide clear, structured outputs.",
        k: int = 3,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        temperature_first: float = 0.0,
        format_prompt: Callable[[str, Any, int, Any], str] = None,
        parse_response: Callable[[str], Any] = None,
        validate_response: Callable[[str, int], bool] = None,
        update_state: Callable[[Any, Any, int], Any] = None,
        initial_state: Any = None,
        max_workers: int = None,
        verbose: bool = True,
        max_retries_per_step: int = 100,
        agents: List[Agent] = None,
    ):
        """
        Initialize the MAKER framework.

        Args:
            id: Unique identifier for the MAKER instance. Auto-generated if not provided.
            name: Human-readable name for the system.
            description: Description of the system's purpose.
            model_name: Name of the LLM model to use (e.g., "gpt-4o-mini", "gpt-4.1-mini").
            system_prompt: System prompt for the microagents. Should describe the task domain
                and expected output format.
            k: Vote threshold - a candidate must be k votes ahead of all others to win.
                Higher k means more reliability but higher cost. Typical values: 2-5.
            max_tokens: Maximum tokens for LLM response. Responses exceeding this are
                red-flagged as the model may be confused.
            temperature: Temperature for LLM sampling (used for votes after the first).
                Lower values (0.1-0.3) provide more consistent results.
            temperature_first: Temperature for first vote. Using 0 ensures the best
                deterministic guess is included in the vote set.
            format_prompt: Function(task, state, step_idx, previous_result) -> str that formats
                the prompt for each step. The task is the main objective passed to run().
                If None, uses a simple default.
            parse_response: Function(response_text) -> result that extracts the result
                from the LLM response. The result must be hashable for voting.
                If None, returns the stripped response text.
            validate_response: Function(response_text, max_tokens) -> bool that validates
                the response format. Returns True if valid, False to red-flag.
                If None, only checks response length.
            update_state: Function(current_state, result, step_idx) -> new_state that
                updates the state after each step. If None, state is unchanged.
            initial_state: Initial state for the task. Can be any type depending on your task.
            max_workers: Maximum parallel workers for concurrent vote sampling.
                If None, uses k as the number of workers.
            verbose: Whether to enable verbose logging.
            max_retries_per_step: Maximum retries per step before raising an error.
            agents: Optional list of pre-configured agents to use instead of creating new ones.
                If provided, agents will be cycled through for each vote.
        """
        self.id = id if id is not None else str(uuid.uuid4())
        self.name = name
        self.description = description
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.k = k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.temperature_first = temperature_first
        self.max_workers = max_workers if max_workers is not None else k
        self.verbose = verbose
        self.max_retries_per_step = max_retries_per_step
        self.agents = agents
        self.initial_state = initial_state

        # Task-specific functions with defaults
        self.format_prompt = (
            format_prompt
            if format_prompt is not None
            else self._default_format_prompt
        )
        self.parse_response = (
            parse_response
            if parse_response is not None
            else self._default_parse_response
        )
        self.validate_response = (
            validate_response
            if validate_response is not None
            else self._default_validate_response
        )
        self.update_state = (
            update_state
            if update_state is not None
            else self._default_update_state
        )

        # Initialize conversation tracker
        self.conversation = Conversation(
            name=f"maker_{self.name}_{self.id}"
        )

        # Statistics tracking
        self.stats = {
            "total_samples": 0,
            "total_votes": 0,
            "red_flagged": 0,
            "steps_completed": 0,
            "votes_per_step": [],
            "samples_per_step": [],
        }

        # Validate configuration
        self._validate_config()

        if self.verbose:
            logger.info(f"MAKER initialized: {self.name}")
            logger.info(
                f"Model: {self.model_name}, k={self.k}, max_tokens={self.max_tokens}"
            )

    def _validate_config(self):
        """
        Validate the MAKER configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.max_tokens < 10:
            raise ValueError("max_tokens must be at least 10")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_retries_per_step < 1:
            raise ValueError("max_retries_per_step must be at least 1")

    def _default_format_prompt(
        self, task: str, state: Any, step_idx: int, previous_result: Any
    ) -> str:
        """
        Default prompt formatter.

        Args:
            task: The main task/objective to complete.
            state: Current state of the task.
            step_idx: Current step index (0-based).
            previous_result: Result from the previous step (None for first step).

        Returns:
            Formatted prompt string.
        """
        prompt_parts = [f"Task: {task}", f"Step {step_idx + 1}:"]

        if state is not None:
            prompt_parts.insert(1, f"Current state: {state}")

        if previous_result is not None:
            prompt_parts.insert(-1, f"Previous result: {previous_result}")

        prompt_parts.append("Provide the result for this step.")

        return "\n".join(prompt_parts)

    def _default_parse_response(self, response_text: str) -> str:
        """
        Default response parser.

        Args:
            response_text: Raw LLM response.

        Returns:
            Stripped response text as the result.
        """
        return response_text.strip()

    def _default_validate_response(
        self, response_text: str, max_tokens: int
    ) -> bool:
        """
        Default response validator (red-flagging).

        Args:
            response_text: Raw LLM response.
            max_tokens: Maximum allowed tokens.

        Returns:
            True if response is valid, False to red-flag.
        """
        # Estimate tokens (rough: 4 chars per token)
        estimated_tokens = len(response_text) // 4

        # Red-flag if too long
        if estimated_tokens > max_tokens:
            return False

        # Red-flag if empty
        if not response_text.strip():
            return False

        return True

    def _default_update_state(
        self, state: Any, result: Any, step_idx: int
    ) -> Any:
        """
        Default state update function (no-op).

        Args:
            state: Current state.
            result: Result from current step.
            step_idx: Current step index.

        Returns:
            Unchanged state.
        """
        return state

    def _create_microagent(self, temperature: float = None) -> Agent:
        """
        Create a focused microagent for a single step.

        Each microagent has minimal context and is focused on solving
        exactly one step of the problem.

        Args:
            temperature: Temperature for this agent's sampling.

        Returns:
            An Agent instance configured for single-step execution.
        """
        temp = temperature if temperature is not None else self.temperature

        agent = Agent(
            agent_name=f"MAKER-MicroAgent-{uuid.uuid4().hex[:8]}",
            agent_description="Focused microagent for single-step execution in MAKER framework",
            system_prompt=self.system_prompt,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=temp,
            max_loops=1,
            verbose=False,
            print_on=False,
            output_type="str-all-except-first",
        )

        return agent

    def _get_agent(self, temperature: float = None) -> Agent:
        """
        Get an agent for voting.

        If agents were provided, returns one from the pool.
        Otherwise, creates a new microagent.

        Args:
            temperature: Temperature for agent sampling.

        Returns:
            An Agent instance.
        """
        if self.agents is not None and len(self.agents) > 0:
            # Cycle through provided agents
            agent_idx = self.stats["total_samples"] % len(self.agents)
            return self.agents[agent_idx]
        else:
            return self._create_microagent(temperature)

    def _make_hashable(self, result: Any) -> Any:
        """
        Convert a result to a hashable type for voting.

        Args:
            result: The result to convert.

        Returns:
            A hashable version of the result.
        """
        if isinstance(result, (str, int, float, bool, type(None))):
            return result
        elif isinstance(result, (list, tuple)):
            return tuple(self._make_hashable(item) for item in result)
        elif isinstance(result, dict):
            return tuple(
                sorted(
                    (k, self._make_hashable(v)) for k, v in result.items()
                )
            )
        elif isinstance(result, set):
            return frozenset(self._make_hashable(item) for item in result)
        else:
            # Fall back to string representation
            return str(result)

    def _unhash_result(self, hashable: Any, original_type: type) -> Any:
        """
        Convert a hashable result back to its original type.

        Args:
            hashable: The hashable result.
            original_type: The original type of the result.

        Returns:
            The result in its original type.
        """
        if original_type in (str, int, float, bool, type(None)):
            return hashable
        elif original_type is list:
            return list(hashable) if isinstance(hashable, tuple) else hashable
        elif original_type is dict:
            return dict(hashable) if isinstance(hashable, tuple) else hashable
        elif original_type is set:
            return set(hashable) if isinstance(hashable, frozenset) else hashable
        else:
            return hashable

    def get_vote(
        self,
        task: str,
        state: Any,
        step_idx: int,
        previous_result: Any = None,
        temperature: float = None,
    ) -> Optional[Tuple[Any, str, type]]:
        """
        Get a single vote for the current step.

        Samples from the LLM and applies red-flagging. If the response has
        red flags, returns None (the vote is discarded).

        This implements Algorithm 3 (get_vote) from the paper.

        Args:
            task: The main task/objective being solved.
            state: Current state of the task.
            step_idx: Current step index.
            previous_result: Result from previous step.
            temperature: Temperature for sampling.

        Returns:
            Tuple of (hashable_result, raw_response, original_type) if valid,
            None if red-flagged.
        """
        self.stats["total_samples"] += 1

        agent = self._get_agent(temperature)
        prompt = self.format_prompt(task, state, step_idx, previous_result)

        try:
            response = agent.run(task=prompt)

            # Red-flag check
            if not self.validate_response(response, self.max_tokens):
                self.stats["red_flagged"] += 1
                if self.verbose:
                    logger.debug(f"Red-flagged response at step {step_idx + 1}")
                return None

            # Parse the response
            result = self.parse_response(response)
            original_type = type(result)

            # Convert to hashable for voting
            hashable_result = self._make_hashable(result)

            self.stats["total_votes"] += 1
            return (hashable_result, response, original_type)

        except Exception as e:
            self.stats["red_flagged"] += 1
            if self.verbose:
                logger.debug(
                    f"Red-flagged response at step {step_idx + 1} (exception: {e})"
                )
            return None

    def do_voting(
        self,
        task: str,
        state: Any,
        step_idx: int,
        previous_result: Any = None,
    ) -> Tuple[Any, str]:
        """
        Perform first-to-ahead-by-k voting for the current step.

        Samples votes until one candidate result is k votes ahead of all others.
        This provides statistical error correction by requiring consensus.

        This implements Algorithm 2 (do_voting) from the paper.

        Args:
            task: The main task/objective being solved.
            state: Current state of the task.
            step_idx: Current step index.
            previous_result: Result from previous step.

        Returns:
            Tuple of (result, raw_response) for the winning candidate.

        Raises:
            RuntimeError: If max_retries_per_step is exceeded without finding a winner.
        """
        votes = {}  # hashable_result -> vote count
        responses = {}  # hashable_result -> raw_response
        original_types = {}  # hashable_result -> original_type
        samples_this_step = 0
        votes_this_step = 0
        is_first_vote = True

        while samples_this_step < self.max_retries_per_step:
            # Use temperature 0 for first vote, then configured temperature
            temp = self.temperature_first if is_first_vote else self.temperature
            is_first_vote = False

            # Get a vote
            result = self.get_vote(task, state, step_idx, previous_result, temp)
            samples_this_step += 1

            if result is None:
                # Red-flagged, try again
                continue

            hashable_result, response, original_type = result
            votes_this_step += 1

            # Update vote count
            if hashable_result not in votes:
                votes[hashable_result] = 0
                responses[hashable_result] = response
                original_types[hashable_result] = original_type
            votes[hashable_result] += 1

            # Check if we have a winner (first-to-ahead-by-k)
            current_count = votes[hashable_result]
            max_other = max(
                (v for r, v in votes.items() if r != hashable_result),
                default=0,
            )

            if current_count >= max_other + self.k:
                # We have a winner!
                self.stats["votes_per_step"].append(votes_this_step)
                self.stats["samples_per_step"].append(samples_this_step)

                if self.verbose:
                    logger.debug(
                        f"Step {step_idx + 1} decided with {votes_this_step} votes "
                        f"({samples_this_step} samples, winner: {current_count} votes)"
                    )

                # Convert back to original type
                final_result = self._unhash_result(
                    hashable_result, original_types[hashable_result]
                )
                return final_result, responses[hashable_result]

        # If we get here, we've exceeded max retries
        raise RuntimeError(
            f"Step {step_idx + 1}: Failed to reach consensus after "
            f"{self.max_retries_per_step} samples. Vote distribution: {votes}"
        )

    def run(self, task: str, max_steps: int = None) -> List[Any]:
        """
        Run the MAKER framework to solve the given task.

        Executes the complete solution process, generating results step-by-step
        using maximal decomposition with error correction through voting.

        This implements Algorithm 1 (generate_solution) from the paper.

        Args:
            task: The main task/objective to complete. This is the primary input
                that defines what the MAKER framework should solve.
            max_steps: Number of steps to execute. Required parameter.

        Returns:
            List of results from each step.

        Raises:
            ValueError: If task is not provided or max_steps is not specified.
            RuntimeError: If voting fails on any step.

        Example:
            >>> maker = MAKER(
            ...     system_prompt="Solve math problems step by step.",
            ...     k=3
            ... )
            >>> results = maker.run(
            ...     task="Calculate 2^10 by doubling, starting from 2",
            ...     max_steps=9
            ... )
        """
        if not task:
            raise ValueError("task is required - this is the objective to complete")
        if max_steps is None:
            raise ValueError("max_steps is required - specify how many steps to execute")

        if self.verbose:
            logger.info(f"Starting MAKER with {max_steps} steps, k={self.k}")
            logger.info(f"Task: {task[:100]}..." if len(task) > 100 else f"Task: {task}")

        # Initialize state
        state = self.initial_state

        results = []
        previous_result = None

        for step_idx in range(max_steps):
            if self.verbose and (step_idx + 1) % max(1, max_steps // 10) == 0:
                logger.info(f"Progress: {step_idx + 1}/{max_steps} steps completed")

            # Do voting for this step
            result, response = self.do_voting(task, state, step_idx, previous_result)

            # Record the result
            results.append(result)

            # Update state
            state = self.update_state(state, result, step_idx)
            previous_result = result

            self.stats["steps_completed"] = step_idx + 1

            # Log to conversation
            self.conversation.add(
                role=f"Step-{step_idx + 1}",
                content=f"Result: {result}",
            )

        if self.verbose:
            self._log_statistics()

        return results

    def run_until_condition(
        self,
        task: str,
        stop_condition: Callable[[Any, List[Any], int], bool],
        max_steps: int = 1000,
    ) -> List[Any]:
        """
        Run MAKER until a stopping condition is met.

        Useful for tasks where the number of steps is not known in advance.

        Args:
            task: The main task/objective to complete.
            stop_condition: Function(current_state, results, step_idx) -> bool
                that returns True when the task is complete.
            max_steps: Maximum steps to prevent infinite loops.

        Returns:
            List of results from each step.

        Example:
            >>> def is_complete(state, results, step_idx):
            ...     return "DONE" in str(results[-1]) if results else False
            >>>
            >>> maker = MAKER(system_prompt="...", k=3)
            >>> results = maker.run_until_condition(
            ...     task="Solve this problem until you reach the answer",
            ...     stop_condition=is_complete,
            ...     max_steps=100
            ... )
        """
        if not task:
            raise ValueError("task is required - this is the objective to complete")
        if stop_condition is None:
            raise ValueError("stop_condition must be provided")

        state = self.initial_state

        if self.verbose:
            logger.info(f"Starting MAKER (conditional), max_steps={max_steps}, k={self.k}")
            logger.info(f"Task: {task[:100]}..." if len(task) > 100 else f"Task: {task}")

        results = []
        previous_result = None

        for step_idx in range(max_steps):
            # Check stop condition
            if stop_condition(state, results, step_idx):
                if self.verbose:
                    logger.info(f"Stop condition met at step {step_idx + 1}")
                break

            if self.verbose and (step_idx + 1) % 10 == 0:
                logger.info(f"Progress: {step_idx + 1} steps completed")

            # Do voting for this step
            result, response = self.do_voting(task, state, step_idx, previous_result)

            results.append(result)
            state = self.update_state(state, result, step_idx)
            previous_result = result
            self.stats["steps_completed"] = step_idx + 1

        if self.verbose:
            self._log_statistics()

        return results

    def run_parallel_voting(self, task: str, max_steps: int = None) -> List[Any]:
        """
        Run MAKER with parallel vote sampling.

        An optimized version that samples k votes in parallel for each step,
        which can significantly reduce wall-clock time while maintaining
        the same error correction guarantees.

        Args:
            task: The main task/objective to complete.
            max_steps: Number of steps to execute.

        Returns:
            List of results from each step.
        """
        if not task:
            raise ValueError("task is required - this is the objective to complete")
        if max_steps is None:
            raise ValueError("max_steps is required - specify how many steps to execute")

        state = self.initial_state

        if self.verbose:
            logger.info(f"Starting MAKER (parallel) with {max_steps} steps, k={self.k}")
            logger.info(f"Task: {task[:100]}..." if len(task) > 100 else f"Task: {task}")

        results = []
        previous_result = None

        for step_idx in range(max_steps):
            if self.verbose and (step_idx + 1) % max(1, max_steps // 10) == 0:
                logger.info(f"Progress: {step_idx + 1}/{max_steps} steps completed")

            result, response = self._do_voting_parallel(
                task, state, step_idx, previous_result
            )

            results.append(result)
            state = self.update_state(state, result, step_idx)
            previous_result = result
            self.stats["steps_completed"] = step_idx + 1

        if self.verbose:
            self._log_statistics()

        return results

    def _do_voting_parallel(
        self,
        task: str,
        state: Any,
        step_idx: int,
        previous_result: Any = None,
    ) -> Tuple[Any, str]:
        """
        Parallel voting implementation.

        Samples k votes in parallel, then continues with sequential sampling
        if no winner is found.

        Args:
            task: The main task/objective being solved.
            state: Current state of the task.
            step_idx: Current step index.
            previous_result: Result from previous step.

        Returns:
            Tuple of (result, raw_response).
        """
        votes = {}
        responses = {}
        original_types = {}
        samples_this_step = 0
        votes_this_step = 0

        # First round: sample k votes in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # First vote with temperature 0, rest with configured temperature
            futures = []
            futures.append(
                executor.submit(
                    self.get_vote,
                    task,
                    state,
                    step_idx,
                    previous_result,
                    self.temperature_first,
                )
            )
            for _ in range(self.k - 1):
                futures.append(
                    executor.submit(
                        self.get_vote,
                        task,
                        state,
                        step_idx,
                        previous_result,
                        self.temperature,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                samples_this_step += 1
                result = future.result()
                if result is not None:
                    hashable_result, response, original_type = result
                    votes_this_step += 1
                    if hashable_result not in votes:
                        votes[hashable_result] = 0
                        responses[hashable_result] = response
                        original_types[hashable_result] = original_type
                    votes[hashable_result] += 1

        # Check if we have a winner, continue sequentially if not
        while samples_this_step < self.max_retries_per_step:
            if votes:
                leader = max(votes, key=votes.get)
                leader_count = votes[leader]
                max_other = max(
                    (v for r, v in votes.items() if r != leader),
                    default=0,
                )

                if leader_count >= max_other + self.k:
                    self.stats["votes_per_step"].append(votes_this_step)
                    self.stats["samples_per_step"].append(samples_this_step)

                    final_result = self._unhash_result(
                        leader, original_types[leader]
                    )
                    return final_result, responses[leader]

            # No winner yet, get more votes sequentially
            result = self.get_vote(
                task, state, step_idx, previous_result, self.temperature
            )
            samples_this_step += 1

            if result is not None:
                hashable_result, response, original_type = result
                votes_this_step += 1
                if hashable_result not in votes:
                    votes[hashable_result] = 0
                    responses[hashable_result] = response
                    original_types[hashable_result] = original_type
                votes[hashable_result] += 1

        raise RuntimeError(
            f"Step {step_idx + 1}: Failed to reach consensus after "
            f"{self.max_retries_per_step} samples"
        )

    def _log_statistics(self):
        """Log execution statistics."""
        logger.info("=" * 50)
        logger.info("MAKER Execution Statistics")
        logger.info("=" * 50)
        logger.info(f"Steps completed: {self.stats['steps_completed']}")
        logger.info(f"Total samples: {self.stats['total_samples']}")
        logger.info(f"Total valid votes: {self.stats['total_votes']}")
        logger.info(f"Red-flagged responses: {self.stats['red_flagged']}")

        if self.stats["votes_per_step"]:
            avg_votes = sum(self.stats["votes_per_step"]) / len(
                self.stats["votes_per_step"]
            )
            max_votes = max(self.stats["votes_per_step"])
            logger.info(f"Average votes per step: {avg_votes:.2f}")
            logger.info(f"Max votes for a step: {max_votes}")

        if self.stats["samples_per_step"]:
            avg_samples = sum(self.stats["samples_per_step"]) / len(
                self.stats["samples_per_step"]
            )
            logger.info(f"Average samples per step: {avg_samples:.2f}")

        red_flag_rate = self.stats["red_flagged"] / max(1, self.stats["total_samples"])
        logger.info(f"Red-flag rate: {red_flag_rate:.2%}")
        logger.info("=" * 50)

    def estimate_cost(
        self, total_steps: int, target_success_probability: float = 0.95
    ) -> Dict[str, Any]:
        """
        Estimate the expected cost of solving a task with given steps.

        Uses the theoretical framework from the paper to estimate costs
        based on step success rate and voting threshold.

        Args:
            total_steps: Total number of steps for the task.
            target_success_probability: Target probability of solving the full task.

        Returns:
            Dictionary containing cost estimates and statistics.
        """
        # Estimate per-step success rate from current statistics
        if self.stats["total_votes"] > 0:
            valid_rate = self.stats["total_votes"] / max(
                1, self.stats["total_samples"]
            )
            p = valid_rate * 0.99  # Assume 99% of valid votes are correct
        else:
            p = 0.99  # Default assumption

        # Calculate minimum k needed (Equation 14 from paper)
        s = total_steps
        t = target_success_probability

        if p > 0.5:
            ratio = (1 - p) / p
            try:
                k_min = math.ceil(math.log(t ** (-1 / s) - 1) / math.log(ratio))
            except (ValueError, ZeroDivisionError):
                k_min = 1
        else:
            k_min = float("inf")

        # Expected samples per step (Equation 16 from paper)
        if p > 0.5 and k_min != float("inf"):
            expected_samples = k_min / (p * (2 * p - 1))
        else:
            expected_samples = float("inf")

        return {
            "estimated_p": p,
            "estimated_k_min": k_min,
            "expected_samples_per_step": expected_samples,
            "expected_total_samples": (
                expected_samples * s
                if expected_samples != float("inf")
                else float("inf")
            ),
            "current_k": self.k,
            "total_steps": s,
            "target_success_probability": t,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary containing execution statistics.
        """
        return self.stats.copy()

    def reset(self):
        """Reset the MAKER instance for a new run."""
        self.stats = {
            "total_samples": 0,
            "total_votes": 0,
            "red_flagged": 0,
            "steps_completed": 0,
            "votes_per_step": [],
            "samples_per_step": [],
        }
        self.conversation = Conversation(name=f"maker_{self.name}_{self.id}")


if __name__ == "__main__":
    import re

    # Example: Using MAKER for a simple step-by-step task
    print("MAKER: General-purpose example")
    print("=" * 50)

    # Define task-specific functions for a counting task
    def format_counting_prompt(task, state, step_idx, previous_result):
        """Format prompt for counting task."""
        if previous_result is None:
            return f"{task}\nThis is step 1. What is the first number? Reply with just the number."
        return f"{task}\nThe previous number was {previous_result}. What is the next number? Reply with just the number."

    def parse_counting_response(response):
        """Parse the counting response to extract the number."""
        numbers = re.findall(r"\d+", response)
        if numbers:
            return int(numbers[0])
        return response.strip()

    def validate_counting_response(response, max_tokens):
        """Validate counting response."""
        if len(response) > max_tokens * 4:
            return False
        return bool(re.search(r"\d+", response))

    # Create MAKER instance
    maker = MAKER(
        name="CountingExample",
        description="MAKER example: counting numbers",
        model_name="gpt-4o-mini",
        system_prompt="You are a helpful assistant. When asked to count, respond with just the number, nothing else.",
        format_prompt=format_counting_prompt,
        parse_response=parse_counting_response,
        validate_response=validate_counting_response,
        k=2,
        max_tokens=100,
        temperature=0.1,
        verbose=True,
    )

    print("\nRunning MAKER to count from 1 to 10...")

    # Run the solver with the task as the main input
    try:
        results = maker.run(
            task="Count from 1 to 10, one number at a time",
            max_steps=10,
        )
        print(f"\nResults: {results}")
        print("Expected: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")

        # Show statistics
        stats = maker.get_statistics()
        print("\nStatistics:")
        print(f"  Steps completed: {stats['steps_completed']}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Red-flagged: {stats['red_flagged']}")
        if stats["votes_per_step"]:
            print(
                f"  Avg votes per step: {sum(stats['votes_per_step'])/len(stats['votes_per_step']):.2f}"
            )
    except Exception as e:
        print(f"Error: {e}")
        print("(This example requires an API key to be configured)")
