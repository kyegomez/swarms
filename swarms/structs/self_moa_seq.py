from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

from loguru import logger
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from swarms.structs.agent import Agent


def retry_with_instance_config(func):
    """
    Decorator that applies retry configuration using instance variables.
    This allows the retry decorator to access instance configuration.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Create retry decorator with instance configuration
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(
                multiplier=self.retry_backoff_multiplier,
                min=self.retry_delay,
                max=self.retry_max_delay,
            ),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=before_sleep_log(logger, "WARNING"),
        )

        # Apply the retry decorator to the function
        retried_func = retry_decorator(func)
        return retried_func(self, *args, **kwargs)

    return wrapper


class SelfMoASeq:
    """
    Self-MoA-Seq: Sequential Self-Mixture of Agents

    An ensemble method that generates multiple outputs from a single
    high-performing model and aggregates them sequentially using a
    sliding window approach. This addresses context length constraints
    while maintaining the effectiveness of in-model diversity.

    Architecture:
    - Phase 1: Generate initial samples from the proposer model
    - Phase 2: Aggregate outputs using sliding window with synthesized bias
    - Phase 3: Iterate until all samples are processed
    """

    def __init__(
        self,
        name: str = "SelfMoASeq",
        description: str = "Self-MoA-Seq: Sequential Self-Mixture of Agents",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        window_size: int = 6,
        reserved_slots: int = 3,
        max_iterations: int = 10,
        max_tokens: int = 2000,
        num_samples: int = 30,
        enable_logging: bool = True,
        log_level: str = "INFO",
        verbose: bool = True,
        proposer_model_name: Optional[str] = None,
        aggregator_model_name: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_multiplier: float = 2.0,
        retry_max_delay: float = 60.0,
        additional_kwargs: Dict[str, Any] = {},
        top_p: Optional[float] = None,
    ):
        # Validate parameters
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if reserved_slots >= window_size:
            raise ValueError(
                "reserved_slots must be less than window_size"
            )
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if retry_backoff_multiplier < 1:
            raise ValueError("retry_backoff_multiplier must be >= 1")
        if retry_max_delay < retry_delay:
            raise ValueError("retry_max_delay must be >= retry_delay")

        # Store parameters
        self.model_name = model_name
        self.temperature = temperature
        self.window_size = window_size
        self.reserved_slots = reserved_slots
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.num_samples = num_samples
        self.enable_logging = enable_logging
        self.log_level = log_level
        self.verbose = verbose

        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.retry_max_delay = retry_max_delay

        # Allow model overrides
        proposer_model = proposer_model_name or self.model_name
        aggregator_model = aggregator_model_name or self.model_name

        # Setup logging
        logger.info(
            f"Initializing Self-MoA-Seq with model: {self.model_name}"
        )

        # Initialize proposer agent (generates multiple samples)
        self.proposer = Agent(
            agent_name="SelfMoASeq-Proposer",
            system_prompt=(
                "You are a sample generator. Generate diverse, high-quality responses "
                "to the given task. Vary your approach while maintaining quality."
            ),
            model_name=proposer_model,
            temperature=self.temperature,
            max_loops=1,
            verbose=self.verbose,
            top_p=top_p,
        )

        # Initialize aggregator agent (synthesizes outputs)
        self.aggregator = Agent(
            agent_name="SelfMoASeq-Aggregator",
            system_prompt=(
                "You are an expert synthesizer. Analyze the provided responses and "
                "synthesize them into a single, high-quality output. Consider the "
                "strengths of each response and combine them effectively. Pay special "
                "attention to any highlighted best response, as it provides high-quality guidance."
            ),
            model_name=aggregator_model,
            temperature=0.0,  # Deterministic aggregation
            max_loops=1,
            verbose=self.verbose,
            top_p=top_p,
        )

        # Metrics tracking
        self.metrics: Dict[str, Any] = {
            "total_samples_generated": 0,
            "total_aggregations": 0,
            "total_tokens_used": 0,
            "execution_time_seconds": 0,
        }

        logger.info("Self-MoA-Seq initialization complete")

    @retry_with_instance_config
    def _generate_samples(
        self, task: str, num_samples: int
    ) -> List[str]:
        """
        Generate multiple samples from the proposer model.

        Args:
            task: The task description
            num_samples: Number of samples to generate

        Returns:
            List of generated samples
        """
        logger.info(f"Generating {num_samples} samples for task")
        samples = []

        try:
            for i in range(num_samples):
                logger.debug(f"Generating sample {i+1}/{num_samples}")
                sample = self.proposer.run(task)
                samples.append(sample)
                self.metrics["total_samples_generated"] += 1

            logger.success(
                f"Successfully generated {len(samples)} samples"
            )
            return samples

        except Exception as e:
            logger.error(f"Error during sample generation: {str(e)}")
            raise

    def _format_aggregation_prompt(
        self,
        task: str,
        samples: List[str],
        best_so_far: Optional[str] = None,
    ) -> str:
        """
        Format the aggregation prompt with sliding window.

        Args:
            task: Original task
            samples: List of samples to aggregate
            best_so_far: Previously synthesized best output

        Returns:
            Formatted aggregation prompt
        """
        prompt = f"Original Task:\n{task}\n\n"

        if best_so_far:
            prompt += f"Current Best Response (synthesized from previous iterations):\n{best_so_far}\n\n"

        prompt += "Candidate Responses to Synthesize:\n"
        for i, sample in enumerate(samples, 1):
            prompt += f"\n[Response {i}]:\n{sample}\n"

        prompt += (
            "\nProvide a comprehensive synthesis that combines the strengths of "
            "all responses while maintaining coherence and quality."
        )

        return prompt

    @retry_with_instance_config
    def _aggregate_window(
        self,
        task: str,
        window_samples: List[str],
        best_so_far: Optional[str] = None,
    ) -> str:
        """
        Aggregate a window of samples.

        Args:
            task: Original task
            window_samples: Samples in current window
            best_so_far: Best aggregation so far

        Returns:
            Synthesized output
        """
        logger.debug(
            f"Aggregating window of {len(window_samples)} samples"
        )

        try:
            prompt = self._format_aggregation_prompt(
                task,
                window_samples,
                best_so_far,
            )

            aggregated = self.aggregator.run(prompt)
            self.metrics["total_aggregations"] += 1

            logger.debug("Window aggregation complete")
            return aggregated

        except Exception as e:
            logger.error(f"Error during window aggregation: {str(e)}")
            raise

    @retry_with_instance_config
    def run(
        self,
        task: str,
    ) -> Dict[str, Any]:
        """
        Execute Self-MoA-Seq on the given task.

        This method implements the sequential aggregation algorithm:
        1. Generate num_samples from the proposer model
        2. Use sliding window to aggregate in chunks
        3. Progressively synthesize outputs, biasing aggregator toward best
        4. Return final synthesized output

        Args:
            task: The task to process

        Returns:
            Dictionary containing:
                - final_output: The best synthesized response
                - all_samples: List of generated samples
                - aggregation_steps: Number of aggregation iterations
                - metrics: Performance metrics
        """
        logger.info(
            f"Starting Self-MoA-Seq run with {self.num_samples} samples"
        )
        start_time = datetime.now()

        try:
            # Validate input
            if not task or not isinstance(task, str):
                raise ValueError("task must be a non-empty string")

            # Phase 1: Generate samples
            logger.info("Phase 1: Generating initial samples")
            samples = self._generate_samples(task, self.num_samples)

            # Phase 2: Sequential aggregation with sliding window
            logger.info("Phase 2: Sequential aggregation")
            best_output = samples[0]
            aggregation_step = 0

            # Process samples in windows
            remaining_samples = samples[1:]

            while remaining_samples:
                aggregation_step += 1
                logger.info(
                    f"Aggregation iteration {aggregation_step}, "
                    f"remaining samples: {len(remaining_samples)}"
                )

                # Create window: reserved slots + new samples
                window_size = min(
                    self.window_size - self.reserved_slots,
                    len(remaining_samples),
                )
                current_window = remaining_samples[:window_size]
                remaining_samples = remaining_samples[window_size:]

                # Aggregate with bias toward best output
                window_with_best = [best_output] + current_window
                best_output = self._aggregate_window(
                    task,
                    window_with_best,
                    best_output,
                )

                if aggregation_step >= self.max_iterations:
                    logger.warning(
                        f"Reached max aggregation iterations ({self.max_iterations})"
                    )
                    break

            # Calculate metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            self.metrics["execution_time_seconds"] = elapsed

            result = {
                "final_output": best_output,
                "all_samples": samples,
                "aggregation_steps": aggregation_step,
                "metrics": self.metrics.copy(),
                "task": task,
                "timestamp": datetime.now().isoformat(),
            }

            logger.success(
                f"Self-MoA-Seq completed in {elapsed:.2f}s "
                f"with {aggregation_step} aggregation iterations"
            )

            if self.verbose:
                self._log_summary(result)

            return result

        except Exception as e:
            logger.error(f"Fatal error in Self-MoA-Seq.run: {str(e)}")
            raise

    def _log_summary(self, result: Dict[str, Any]) -> None:
        """Log execution summary."""
        logger.info("=" * 60)
        logger.info("Self-MoA-Seq Execution Summary")
        logger.info("=" * 60)
        logger.info(
            f"Total samples generated: {self.metrics['total_samples_generated']}"
        )
        logger.info(
            f"Aggregation iterations: {result['aggregation_steps']}"
        )
        logger.info(
            f"Execution time: {self.metrics['execution_time_seconds']:.2f}s"
        )
        logger.info(
            f"Final output length: {len(result['final_output'])} chars"
        )
        logger.info("=" * 60)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()
