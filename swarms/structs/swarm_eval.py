import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional, Tuple, List

from datasets import Dataset, load_dataset
from loguru import logger
from tqdm import tqdm
from swarms.security import SwarmShieldIntegration, ShieldConfig

# -----------------------------------------------------------------------------
# Logging configuration: log to console and file (rotating by size)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Swarm interface example
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Benchmark configuration
# -----------------------------------------------------------------------------
class BenchmarkConfig:
    """
    Configuration for a benchmark dataset.

    Attributes:
        input_column (str): The column containing the task prompt.
        answer_column (str): The column containing the expected answer.
        answer_extractor (Optional[Callable[[Any], str]]): Function to extract
            a string answer from the dataset's raw answer format.
        answer_matcher (Optional[Callable[[str, str], bool]]): Function to compare
            the expected answer and the swarm output. If None, a simple substring
            containment is used.
    """

    def __init__(
        self,
        input_column: str,
        answer_column: str,
        answer_extractor: Optional[Callable[[Any], str]] = None,
        answer_matcher: Optional[Callable[[str, str], bool]] = None,
    ):
        self.input_column = input_column
        self.answer_column = answer_column
        self.answer_extractor = answer_extractor
        self.answer_matcher = answer_matcher


# -----------------------------------------------------------------------------
# Preset dataset configurations for popular benchmarks
# -----------------------------------------------------------------------------
PRESET_DATASETS: Dict[str, BenchmarkConfig] = {
    "gsm8k": BenchmarkConfig(
        input_column="question",
        answer_column="answer",
    ),
    "squad": BenchmarkConfig(
        input_column="question",
        answer_column="answers",
        answer_extractor=lambda ans: (
            ans["text"][0]
            if isinstance(ans, dict)
            and "text" in ans
            and isinstance(ans["text"], list)
            and ans["text"]
            else str(ans)
        ),
    ),
    "winogrande": BenchmarkConfig(
        input_column="sentence",
        answer_column="answer",
    ),
    "commonsense_qa": BenchmarkConfig(
        input_column="question",
        answer_column="answerKey",
    ),
    # Add additional presets here.
}


# -----------------------------------------------------------------------------
# SwarmEvaluator with extended features
# -----------------------------------------------------------------------------
class SwarmEvaluator:
    """
    Evaluator that uses a swarm of agents to process benchmark datasets
    from Hugging Face, with concurrency, retries, progress display, performance timing,
    and customizable answer matching.

    Example:
        swarm = Swarm()
        evaluator = SwarmEvaluator(swarm)
        results = evaluator.evaluate("gsm8k", split="test", max_workers=4)
        print(results)
    """

    def __init__(
        self, 
        swarm: callable,
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard",
    ) -> None:
        """
        Initialize the evaluator with a given swarm.

        Args:
            swarm (Swarm): A swarm instance with a callable run(task: str) method.
            shield_config (ShieldConfig, optional): Security configuration for SwarmShield integration. Defaults to None.
            enable_security (bool, optional): Whether to enable SwarmShield security features. Defaults to True.
            security_level (str, optional): Pre-defined security level. Options: "basic", "standard", "enhanced", "maximum". Defaults to "standard".
        """
        self.swarm = swarm

        # Initialize SwarmShield integration
        self._initialize_swarm_shield(shield_config, enable_security, security_level)

    def _initialize_swarm_shield(
        self, 
        shield_config: Optional[ShieldConfig] = None,
        enable_security: bool = True,
        security_level: str = "standard"
    ) -> None:
        """Initialize SwarmShield integration for security features."""
        self.enable_security = enable_security
        self.security_level = security_level
        
        if enable_security:
            if shield_config is None:
                shield_config = ShieldConfig.get_security_level(security_level)
            
            self.swarm_shield = SwarmShieldIntegration(shield_config)
        else:
            self.swarm_shield = None

    # Security methods
    def validate_task_with_shield(self, task: str) -> str:
        """Validate and sanitize task input using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.validate_and_protect_input(task)
        return task

    def validate_agent_config_with_shield(self, agent_config: dict) -> dict:
        """Validate agent configuration using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.validate_and_protect_input(str(agent_config))
        return agent_config

    def process_agent_communication_with_shield(self, message: str, agent_name: str) -> str:
        """Process agent communication through SwarmShield security."""
        if self.swarm_shield:
            return self.swarm_shield.process_agent_communication(message, agent_name)
        return message

    def check_rate_limit_with_shield(self, agent_name: str) -> bool:
        """Check rate limits for an agent using SwarmShield."""
        if self.swarm_shield:
            return self.swarm_shield.check_rate_limit(agent_name)
        return True

    def add_secure_message(self, message: str, agent_name: str) -> None:
        """Add a message to secure conversation history."""
        if self.swarm_shield:
            self.swarm_shield.add_secure_message(message, agent_name)

    def get_secure_messages(self) -> List[dict]:
        """Get secure conversation messages."""
        if self.swarm_shield:
            return self.swarm_shield.get_secure_messages()
        return []

    def get_security_stats(self) -> dict:
        """Get security statistics and metrics."""
        if self.swarm_shield:
            return self.swarm_shield.get_security_stats()
        return {"security_enabled": False}

    def update_shield_config(self, new_config: ShieldConfig) -> None:
        """Update SwarmShield configuration."""
        if self.swarm_shield:
            self.swarm_shield.update_config(new_config)

    def enable_security(self) -> None:
        """Enable SwarmShield security features."""
        if not self.swarm_shield:
            self._initialize_swarm_shield(enable_security=True, security_level=self.security_level)

    def disable_security(self) -> None:
        """Disable SwarmShield security features."""
        self.swarm_shield = None
        self.enable_security = False

    def cleanup_security(self) -> None:
        """Clean up SwarmShield resources."""
        if self.swarm_shield:
            self.swarm_shield.cleanup()

    def evaluate(
        self,
        dataset_name: str,
        split: str = "test",
        config: Optional[BenchmarkConfig] = None,
        max_workers: int = 1,
        max_retries: int = 3,
        show_progress: bool = True,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the specified benchmark dataset using the swarm.

        Args:
            dataset_name (str): The dataset name (from Hugging Face).
            split (str): The dataset split (e.g., "test", "validation").
            config (Optional[BenchmarkConfig]): Benchmark configuration. If None,
                a preset config is used.
            max_workers (int): Number of concurrent workers.
            max_retries (int): Number of retries for swarm tasks on failure.
            show_progress (bool): If True, display a progress bar.
            output_file (Optional[str]): Path to a file to write the results.

        Returns:
            Dict[str, Any]: Evaluation metrics including total examples, correct answers,
            accuracy, and total evaluation time.
        """
        if config is None:
            config = PRESET_DATASETS.get(dataset_name)
            if config is None:
                raise ValueError(
                    f"No preset config for dataset '{dataset_name}'. Provide a BenchmarkConfig."
                )

        logger.info(
            f"Loading dataset '{dataset_name}' (split: {split})..."
        )
        dataset: Dataset = load_dataset(dataset_name, split=split)
        total_examples = len(dataset)
        logger.info(f"Total examples to evaluate: {total_examples}")

        start_time = time.time()
        correct = 0

        # Function to process a single example.
        def _process_example(
            example: Dict[str, Any], idx: int
        ) -> Tuple[bool, float]:
            task_start = time.time()
            task_text = example.get(config.input_column)
            expected_answer = example.get(config.answer_column)

            if task_text is None or expected_answer is None:
                logger.warning(
                    f"Example {idx}: Missing '{config.input_column}' or '{config.answer_column}', skipping."
                )
                return (False, 0.0)

            # Use answer_extractor if provided.
            if config.answer_extractor:
                try:
                    expected_answer = config.answer_extractor(
                        expected_answer
                    )
                except Exception as e:
                    logger.error(
                        f"Example {idx}: Error extracting answer: {e}"
                    )
                    return (False, 0.0)

            logger.debug(f"Example {idx} - Task: {task_text}")
            logger.debug(
                f"Example {idx} - Expected Answer: {expected_answer}"
            )

            try:
                swarm_output = self._run_with_retry(
                    task_text, max_retries
                )
            except Exception as e:
                logger.error(
                    f"Example {idx}: Failed after retries. Error: {e}"
                )
                return (False, time.time() - task_start)

            logger.debug(
                f"Example {idx} - Swarm Output: {swarm_output}"
            )

            # Use custom matcher if provided; otherwise, default matching.
            if config.answer_matcher:
                is_correct = config.answer_matcher(
                    expected_answer, swarm_output
                )
            else:
                is_correct = self._default_matcher(
                    expected_answer, swarm_output
                )

            task_time = time.time() - task_start
            logger.info(
                f"Example {idx}: {'Correct' if is_correct else 'Incorrect'} in {task_time:.2f}s"
            )
            return (is_correct, task_time)

        # Use ThreadPoolExecutor for concurrency.
        futures = []
        total_time = 0.0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Optionally wrap the dataset with tqdm for a progress bar.
            examples_iter = enumerate(dataset, start=1)
            if show_progress:
                examples_iter = tqdm(
                    list(examples_iter),
                    total=total_examples,
                    desc="Evaluating",
                )

            for idx, example in examples_iter:
                futures.append(
                    executor.submit(_process_example, example, idx)
                )

            for future in as_completed(futures):
                try:
                    is_correct, elapsed = future.result()
                    total_time += elapsed
                    if is_correct:
                        correct += 1
                except Exception as e:
                    logger.error(f"Error processing an example: {e}")

        overall_time = time.time() - start_time
        accuracy = (
            correct / total_examples if total_examples > 0 else 0.0
        )

        logger.info(
            f"Evaluation complete. Total examples: {total_examples}, Correct: {correct}, "
            f"Accuracy: {accuracy:.2%}, Overall Time: {overall_time:.2f}s, "
            f"Average per-example time: {total_time/total_examples if total_examples else 0:.2f}s"
        )

        results = {
            "total": total_examples,
            "correct": correct,
            "accuracy": accuracy,
            "overall_time": overall_time,
            "average_example_time": (
                total_time / total_examples
                if total_examples
                else math.nan
            ),
        }

        # Optionally save results to a file.
        if output_file:
            try:
                with open(output_file, "w") as f:
                    for key, value in results.items():
                        f.write(f"{key}: {value}\n")
                logger.info(f"Results saved to {output_file}")
            except Exception as e:
                logger.error(
                    f"Error saving results to {output_file}: {e}"
                )

        return results

    def _run_with_retry(self, task: str, max_retries: int) -> str:
        """
        Runs the swarm task with a retry mechanism.

        Args:
            task (str): The task string.
            max_retries (int): Maximum number of retries.

        Returns:
            str: Swarm output.

        Raises:
            Exception: If all retries fail.
        """
        attempt = 0
        while attempt <= max_retries:
            try:
                start = time.time()
                result = self.swarm.run(task)
                elapsed = time.time() - start
                logger.debug(
                    f"Task succeeded in {elapsed:.2f}s on attempt {attempt + 1}"
                )
                return result
            except Exception as e:
                logger.warning(
                    f"Task failed on attempt {attempt + 1}: {e}"
                )
                attempt += 1
                time.sleep(0.5 * attempt)  # Exponential backoff
        raise Exception("Max retries exceeded for task.")

    @staticmethod
    def _default_matcher(expected: str, output: str) -> bool:
        """
        Default answer matching using a normalized substring check.

        Args:
            expected (str): The expected answer.
            output (str): The swarm output.

        Returns:
            bool: True if expected is found in output; otherwise, False.
        """
        expected_norm = " ".join(expected.strip().split())
        output_norm = " ".join(output.strip().split())
        return expected_norm in output_norm


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
