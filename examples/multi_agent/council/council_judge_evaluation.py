import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from swarms.structs.agent import Agent
from swarms.structs.council_judge import CouncilAsAJudge

# Dataset configurations
DATASET_CONFIGS = {
    "gsm8k": "main",
    "squad": None,  # No specific config needed
    "winogrande": None,
    "commonsense_qa": None,
}


base_agent = Agent(
    agent_name="General-Problem-Solver",
    system_prompt="""You are an expert problem solver and analytical thinker with deep expertise across multiple domains. Your role is to break down complex problems, identify key patterns, and provide well-reasoned solutions.

Key Responsibilities:
1. Analyze problems systematically by breaking them into manageable components
2. Identify relevant patterns, relationships, and dependencies
3. Apply logical reasoning and critical thinking to evaluate solutions
4. Consider multiple perspectives and potential edge cases
5. Provide clear, step-by-step explanations of your reasoning
6. Validate solutions against given constraints and requirements

Problem-Solving Framework:
1. Problem Understanding
   - Identify the core problem and key objectives
   - Clarify constraints and requirements
   - Define success criteria

2. Analysis
   - Break down complex problems into components
   - Identify relevant patterns and relationships
   - Consider multiple perspectives and approaches

3. Solution Development
   - Generate potential solutions
   - Evaluate trade-offs and implications
   - Select optimal approach based on criteria

4. Validation
   - Test solution against requirements
   - Consider edge cases and potential issues
   - Verify logical consistency

5. Communication
   - Present clear, structured reasoning
   - Explain key decisions and trade-offs
   - Provide actionable recommendations

Remember to maintain a systematic, analytical approach while being adaptable to different problem domains.""",
    model_name="gpt-4o-mini",
    max_loops=1,
    max_tokens=16000,
)


class CouncilJudgeEvaluator:
    """
    Evaluates the Council of Judges using various datasets from Hugging Face.
    Checks if the council's output contains the correct answer from the dataset.
    """

    def __init__(
        self,
        base_agent: Optional[Agent] = base_agent,
        model_name: str = "gpt-4o-mini",
        output_dir: str = "evaluation_results",
    ):
        """
        Initialize the Council Judge Evaluator.

        Args:
            base_agent: Optional base agent to use for responses
            model_name: Model to use for evaluations
            output_dir: Directory to save evaluation results
        """

        self.council = CouncilAsAJudge(
            base_agent=base_agent,
            output_type="final",
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load existing results
        self.results_file = (
            self.output_dir / "evaluation_results.json"
        )
        self.results = self._load_or_create_results()

    def _load_or_create_results(self) -> Dict[str, Any]:
        """Load existing results or create new results structure."""
        if self.results_file.exists():
            try:
                with open(self.results_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    "Existing results file is corrupted. Creating new one."
                )

        return {
            "datasets": {},
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_evaluations": 0,
            "total_correct": 0,
        }

    def _save_results(self):
        """Save current results to file."""
        self.results["last_updated"] = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {self.results_file}")

    def evaluate_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        num_samples: Optional[int] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate the Council of Judges on a specific dataset.

        Args:
            dataset_name: Name of the Hugging Face dataset
            split: Dataset split to use
            num_samples: Number of samples to evaluate (None for all)
            save_results: Whether to save results to file

        Returns:
            Dictionary containing evaluation metrics and results
        """
        logger.info(
            f"Loading dataset {dataset_name} (split: {split})..."
        )

        # Get dataset config if needed
        config = DATASET_CONFIGS.get(dataset_name)
        if config:
            dataset = load_dataset(dataset_name, config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)

        if num_samples:
            dataset = dataset.select(
                range(min(num_samples, len(dataset)))
            )

        # Initialize or get existing dataset results
        if dataset_name not in self.results["datasets"]:
            self.results["datasets"][dataset_name] = {
                "evaluations": [],
                "correct_answers": 0,
                "total_evaluated": 0,
                "accuracy": 0.0,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

        start_time = time.time()

        for idx, example in enumerate(
            tqdm(dataset, desc="Evaluating samples")
        ):
            try:
                # Get the input text and correct answer based on dataset structure
                input_text = self._get_input_text(
                    example, dataset_name
                )
                correct_answer = self._get_correct_answer(
                    example, dataset_name
                )

                # Run evaluation through council
                evaluation = self.council.run(input_text)

                # Check if the evaluation contains the correct answer
                is_correct = self._check_answer(
                    evaluation, correct_answer, dataset_name
                )

                # Create sample result
                sample_result = {
                    "input": input_text,
                    "correct_answer": correct_answer,
                    "evaluation": evaluation,
                    "is_correct": is_correct,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                # Update dataset results
                self.results["datasets"][dataset_name][
                    "evaluations"
                ].append(sample_result)
                if is_correct:
                    self.results["datasets"][dataset_name][
                        "correct_answers"
                    ] += 1
                    self.results["total_correct"] += 1
                self.results["datasets"][dataset_name][
                    "total_evaluated"
                ] += 1
                self.results["total_evaluations"] += 1

                # Update accuracy
                self.results["datasets"][dataset_name]["accuracy"] = (
                    self.results["datasets"][dataset_name][
                        "correct_answers"
                    ]
                    / self.results["datasets"][dataset_name][
                        "total_evaluated"
                    ]
                )
                self.results["datasets"][dataset_name][
                    "last_updated"
                ] = time.strftime("%Y-%m-%d %H:%M:%S")

                # Save results after each evaluation
                if save_results:
                    self._save_results()

            except Exception as e:
                logger.error(
                    f"Error evaluating sample {idx}: {str(e)}"
                )
                continue

        # Calculate final metrics
        results = {
            "dataset": dataset_name,
            "split": split,
            "num_samples": len(dataset),
            "evaluations": self.results["datasets"][dataset_name][
                "evaluations"
            ],
            "correct_answers": self.results["datasets"][dataset_name][
                "correct_answers"
            ],
            "total_evaluated": self.results["datasets"][dataset_name][
                "total_evaluated"
            ],
            "accuracy": self.results["datasets"][dataset_name][
                "accuracy"
            ],
            "total_time": time.time() - start_time,
        }

        return results

    def _get_input_text(
        self, example: Dict, dataset_name: str
    ) -> str:
        """Extract input text based on dataset structure."""
        if dataset_name == "gsm8k":
            return example["question"]
        elif dataset_name == "squad":
            return example["question"]
        elif dataset_name == "winogrande":
            return example["sentence"]
        elif dataset_name == "commonsense_qa":
            return example["question"]
        else:
            # Default to first field that looks like text
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 10:
                    return value
            raise ValueError(
                f"Could not find input text in example for dataset {dataset_name}"
            )

    def _get_correct_answer(
        self, example: Dict, dataset_name: str
    ) -> str:
        """Extract correct answer based on dataset structure."""
        if dataset_name == "gsm8k":
            return str(example["answer"])
        elif dataset_name == "squad":
            return (
                example["answers"]["text"][0]
                if isinstance(example["answers"], dict)
                else str(example["answers"])
            )
        elif dataset_name == "winogrande":
            return str(example["answer"])
        elif dataset_name == "commonsense_qa":
            return str(example["answerKey"])
        else:
            # Try to find an answer field
            for key in ["answer", "answers", "label", "target"]:
                if key in example:
                    return str(example[key])
            raise ValueError(
                f"Could not find correct answer in example for dataset {dataset_name}"
            )

    def _check_answer(
        self, evaluation: str, correct_answer: str, dataset_name: str
    ) -> bool:
        """Check if the evaluation contains the correct answer."""
        # Convert both to lowercase for case-insensitive comparison
        evaluation_lower = evaluation.lower()
        correct_answer_lower = correct_answer.lower()

        # For GSM8K, we need to extract the final numerical answer
        if dataset_name == "gsm8k":
            try:
                # Look for the final answer in the format "The answer is X" or "Answer: X"
                import re

                final_answer = re.search(
                    r"(?:the answer is|answer:)\s*(\d+)",
                    evaluation_lower,
                )
                if final_answer:
                    return (
                        final_answer.group(1) == correct_answer_lower
                    )
            except:
                pass

        # For other datasets, check if the correct answer is contained in the evaluation
        return correct_answer_lower in evaluation_lower


def main():
    # Example usage
    evaluator = CouncilJudgeEvaluator()

    # Evaluate on multiple datasets
    datasets = ["gsm8k", "squad", "winogrande", "commonsense_qa"]

    for dataset in datasets:
        try:
            logger.info(f"\nEvaluating on {dataset}...")
            results = evaluator.evaluate_dataset(
                dataset_name=dataset,
                split="test",
                num_samples=10,  # Limit samples for testing
            )

            # Print summary
            print(f"\nResults for {dataset}:")
            print(f"Accuracy: {results['accuracy']:.3f}")
            print(
                f"Correct answers: {results['correct_answers']}/{results['total_evaluated']}"
            )
            print(f"Total time: {results['total_time']:.2f} seconds")

        except Exception as e:
            logger.error(f"Error evaluating {dataset}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
