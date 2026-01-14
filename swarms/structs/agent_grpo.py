import uuid
import math
from typing import List, Union, Callable, Any, Dict, Optional
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_exec import run_agents_concurrently


class AgenticGRPO:
    """
    Agentic Group Relative Policy Optimization (GRPO) implementation.

    This class manages agent sampling, response rating, and tracks responses
    with unique identifiers, tasks, answers, and ratings.
    """

    def __init__(
        self,
        name: str,
        description: str,
        agent: Union[Agent, Callable] = None,
        n: int = 10,
        correct_answers: Optional[List[str]] = None,
    ):
        """
        Initialize the AgenticGRPO instance.

        Args:
            name: Name of the GRPO instance
            description: Description of the GRPO instance
            agent: Agent instance or callable to use for sampling
            n: Number of samples to generate
            correct_answers: Optional list of strings representing correct answers to check against
        """
        self.name = name
        self.description = description
        self.agent = agent
        self.n = n
        self.correct_answers = (
            correct_answers if correct_answers is not None else []
        )
        self.responses_hashmap = []

    def sample(self, task: str = None):
        """
        Sample responses from the agent.

        Args:
            task: Task string to pass to the agent

        Returns:
            List of responses from the agent
        """
        # Create n copies of the agent for concurrent execution
        agents = [self.agent] * self.n

        return run_agents_concurrently(
            agents=agents,
            task=task,
        )

    def rate_answers_to_correct_answer(
        self, responses: List[str], task: str = None
    ):
        """
        Rate all answers against the list of correct answers and add to responses_hashmap.

        Args:
            responses: List of response strings to rate
            task: The task that was used to generate these responses
        """
        for response in responses:
            response_id = str(uuid.uuid4())
            rating = self.rate_answer_to_correct_answer(response)

            response_entry = {
                "uuid": response_id,
                "task": task,
                "answer": response,
                "rating": rating,
                "advantage": None,  # Will be computed later
            }

            self.responses_hashmap.append(response_entry)

    def rate_answer_to_correct_answer(self, response: str):
        """
        Rate a single answer against the list of correct answers.

        Args:
            response: The response string to rate

        Returns:
            1 if any correct answer is found in response, 0 otherwise
        """
        if not self.correct_answers:
            return 0

        response_str = str(response)
        for correct_answer in self.correct_answers:
            if str(correct_answer) in response_str:
                return 1
        return 0

    def compute_group_baseline(self) -> float:
        """
        Compute the group baseline (average rating) for all responses.

        Formula: r_bar = (1/K) * sum(r_i from i=1 to K)
        where K is the number of completions and r_i is the rating for the i-th completion.

        Returns:
            Average rating (group baseline)
        """
        if not self.responses_hashmap:
            return 0.0

        total_rating = sum(
            response.get("rating", 0)
            for response in self.responses_hashmap
        )
        k = len(self.responses_hashmap)
        return total_rating / k if k > 0 else 0.0

    def compute_advantages(self, normalize: bool = False):
        """
        Compute relative advantages for all responses (GRPO style).

        Step 1: Compute group baseline: r_bar = (1/K) * sum(r_i)
        Step 2: Compute advantage for each completion: A_i = r_i - r_bar
        Step 3 (optional): Normalize: A_i <- (A_i - mu) / sigma

        This is crucial:
        - Multiple correct answers all get positive reinforcement
        - Worse completions get negative reinforcement
        - Neutral completions (around average) don't move the model much

        Args:
            normalize: Whether to normalize the advantages (default: False)
        """
        if not self.responses_hashmap:
            return

        # Step 1: Compute group baseline
        r_bar = self.compute_group_baseline()

        # Step 2: Compute advantage for each completion
        advantages = []
        for response in self.responses_hashmap:
            r_i = response.get("rating", 0)
            A_i = r_i - r_bar
            response["advantage"] = A_i
            advantages.append(A_i)

        # Step 3: Optional normalization
        if normalize and len(advantages) > 1:
            # Compute mean and standard deviation of advantages
            mu = sum(advantages) / len(advantages)
            variance = sum((a - mu) ** 2 for a in advantages) / len(
                advantages
            )
            sigma = math.sqrt(variance) if variance > 0 else 1.0

            # Normalize each advantage
            for response in self.responses_hashmap:
                if response["advantage"] is not None:
                    A_i = response["advantage"]
                    response["advantage"] = (
                        (A_i - mu) / sigma if sigma > 0 else 0.0
                    )

    def get_correct_responses(self) -> List[Dict[str, Any]]:
        """
        Return all responses that got the correct answer (rating == 1).

        Returns:
            List of dictionaries containing uuid, task, answer, and rating
            for all responses with rating == 1
        """
        return [
            response
            for response in self.responses_hashmap
            if response.get("rating") == 1
        ]

    def run(
        self, task: str = None, normalize_advantages: bool = False
    ):
        """
        Run the sampling, rating, and advantage computation process.

        Args:
            task: Task string to pass to the agent
            normalize_advantages: Whether to normalize the computed advantages (default: False)

        Returns:
            List of response entries (dictionaries with uuid, task, answer, rating, advantage)
            for all responses that got the correct answer (rating == 1)
        """
        samples = self.sample(task)
        self.rate_answers_to_correct_answer(samples, task)
        self.compute_advantages(normalize=normalize_advantages)
        return self.get_correct_responses()

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Return all responses from the responses_hashmap.

        Returns:
            List of dictionaries containing uuid, task, answer, rating, and advantage
            for all responses
        """
        return self.responses_hashmap

    def get_group_baseline(self) -> float:
        """
        Get the current group baseline (average rating).

        Returns:
            The group baseline value
        """
        return self.compute_group_baseline()
