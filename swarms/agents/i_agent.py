"""
Iterative Reflective Expansion (IRE) Algorithm

A sophisticated reasoning framework that employs iterative hypothesis generation, simulation, and refinement to solve complex problems. IRE leverages a multi-step approach where an AI agent generates initial solution paths, evaluates their effectiveness through simulation, reflects on errors, and dynamically revises reasoning strategies. Through continuous cycles of hypothesis testing and meta-cognitive reflection, the algorithm progressively converges on optimal solutions by learning from both successful and unsuccessful reasoning attempts.


- IRE is a multi-step approach where an AI agent generates initial solution paths, evaluates their effectiveness through simulation, reflects on errors, and dynamically revises reasoning strategies.
- Through continuous cycles of hypothesis testing and meta-cognitive reflection, the algorithm progressively converges on optimal solutions by learning from both successful and unsuccessful reasoning attempts.


Workflow:
1. Generate initial hypotheses
2. Simulate paths
3. Reflect on errors
4. Revise paths
5. Select promising paths
6. Synthesize solution

"""

import re
from typing import List, Tuple
from loguru import logger
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.output_types import OutputType
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)

# Define a new system prompt for general problem solving
GENERAL_REASONING_AGENT_SYS_PROMPT = """
You are a highly capable problem-solving agent with a unique ability to reason through complex challenges via iterative reflection and hypothesis testing.
Your role is to assist in generating innovative solutions to a wide array of general problems by engaging in trial and error, reflective evaluation, and dynamic hypothesis expansion.
When presented with a problem statement, generate multiple hypotheses, simulate reasoning paths, reflect on errors, and iteratively refine your approach to produce the best solution.
Do not include any finance-related content.

"""

# Configuration constants
MAX_PATHS_PER_ITERATION = 5
SCORE_THRESHOLD = 0.7
EARLY_TERMINATION_SCORE = 0.85
DEFAULT_SCORE = 0.5


class IterativeReflectiveExpansion:
    """
    A class implementing the Iterative Reflective Expansion (IRE) reasoning algorithm.

    This algorithm leverages a Swarms agent to iteratively generate, simulate, reflect on, and refine reasoning paths
    in order to solve complex problems through trial and error, reflective evaluation, and dynamic hypothesis expansion.
    """

    def __init__(
        self,
        agent_name: str = "General-Reasoning-Agent",
        description: str = "A reasoning agent that can answer questions and help with tasks.",
        agent: Agent = None,
        max_iterations: int = 5,
        system_prompt: str = GENERAL_REASONING_AGENT_SYS_PROMPT,
        model_name: str = "gpt-4o-mini",
        output_type: OutputType = "dict",
    ) -> None:
        """
        Initialize the Iterative Reflective Expansion engine.

        :param agent: The Swarms agent instance used to perform reasoning tasks.
        :param max_iterations: Maximum number of iterations for the reasoning process.
        """
        self.agent_name = agent_name
        self.description = description
        self.agent = agent
        self.max_iterations = max_iterations
        self.output_type = output_type
        self.system_prompt = system_prompt
        self.conversation = Conversation()

        self.agent = Agent(
            agent_name=self.agent_name,
            system_prompt=self.system_prompt,
            model_name=model_name,
            max_loops=1,
            dynamic_temperature_enabled=True,
        )

    def _extract_score_robust(self, response: str) -> float:
        """
        Robustly extract a score from LLM response using multiple strategies.

        :param response: The LLM response text.
        :return: Extracted score between 0.0 and 1.0, or DEFAULT_SCORE if extraction fails.
        """
        # Strategy 1: Look for "Score: X.X" format (with or without markdown formatting)
        for line in response.splitlines():
            line_clean = line.strip().replace('*', '')  # Remove markdown formatting
            if 'score:' in line_clean.lower():
                try:
                    # Extract everything after "score:"
                    score_str = line_clean.lower().split('score:')[-1].strip()
                    # Remove any non-numeric characters except decimal point
                    score_str = re.sub(r'[^\d.]', '', score_str)
                    if score_str:  # Make sure we have something to parse
                        score = float(score_str)
                        # Clamp to valid range
                        return max(0.0, min(1.0, score))
                except (ValueError, IndexError):
                    pass

        # Strategy 2: Look for any number between 0 and 1 with context
        score_patterns = [
            r'score[:\s]+(\d+\.?\d*)',
            r'rating[:\s]+(\d+\.?\d*)',
            r'effectiveness[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:/|out of)\s*(?:10|1\.0|1)',
        ]

        for pattern in score_patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                try:
                    score = float(matches[0])
                    # Normalize if score is out of 10
                    if score > 1.0:
                        score = score / 10.0
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue

        # Strategy 3: Sentiment analysis fallback
        positive_keywords = ['excellent', 'good', 'promising', 'effective', 'successful', 'optimal']
        negative_keywords = ['poor', 'bad', 'ineffective', 'failed', 'error', 'wrong', 'incorrect']

        response_lower = response.lower()
        positive_count = sum(1 for kw in positive_keywords if kw in response_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in response_lower)

        if positive_count > negative_count and positive_count > 0:
            return 0.75  # Likely good
        elif negative_count > positive_count and negative_count > 0:
            return 0.4   # Likely poor

        # Default fallback
        logger.warning(f"Could not extract score from response, using default: {DEFAULT_SCORE}")
        return DEFAULT_SCORE

    def generate_initial_hypotheses(self, task: str) -> List[str]:
        """
        Generate an initial set of reasoning hypotheses based on the problem input.

        :param task: The problem statement.
        :return: A list of candidate reasoning paths/hypotheses.
        """
        logger.info("Generating initial hypotheses for the problem.")
        prompt = (
            f"Given the following problem:\n\n"
            f"'{task}'\n\n"
            "Generate a list of possible approaches and strategies to solve it. "
            "Present each approach on a new line."
        )
        response = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=response
        )
        hypotheses = [
            line.strip()
            for line in response.split("\n")
            if line.strip()
        ]
        logger.debug(f"Initial hypotheses: {hypotheses}")
        return hypotheses

    def simulate_path(self, path: str) -> Tuple[str, float, str]:
        """
        Simulate a given reasoning path and evaluate its effectiveness.

        :param path: A candidate reasoning path.
        :return: A tuple containing the simulated outcome, a numerical score (0.0 to 1.0), and error information.
        """
        logger.info(f"Simulating path: {path[:100]}...")
        prompt = (
            f"Simulate the following reasoning path step by step and provide:\n"
            f"1. Outcome: A brief summary of the resulting solution.\n"
            f"2. Score: A numerical effectiveness score between 0.0 and 1.0 (REQUIRED - provide a decimal number).\n"
            f"3. Errors: Any potential errors or shortcomings identified during the reasoning.\n\n"
            f"IMPORTANT: You MUST provide a score as a decimal number (e.g., 0.8, 0.65, 0.9).\n\n"
            f"Reasoning Path: {path}"
        )
        response = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=response
        )

        outcome = ""
        error_info = ""

        # Extract outcome and errors (handle markdown formatting)
        for line in response.splitlines():
            line_stripped = line.strip().replace('*', '')  # Remove markdown
            line_lower = line_stripped.lower()

            if 'outcome:' in line_lower:
                outcome = line_stripped.split(':', 1)[-1].strip()
            elif 'errors:' in line_lower or 'error:' in line_lower:
                error_info = line_stripped.split(':', 1)[-1].strip()

        # Use robust score extraction
        score = self._extract_score_robust(response)

        # If no explicit errors found, check for error indicators in outcome
        if not error_info and outcome:
            error_keywords = ['error', 'fail', 'incorrect', 'wrong', 'issue', 'problem']
            if any(kw in outcome.lower() for kw in error_keywords):
                error_info = "Potential issues identified in outcome"

        logger.info(f"Path score: {score:.2f} | Outcome length: {len(outcome)} chars")
        return outcome, score, error_info

    def meta_reflect(self, error_info: str) -> str:
        """
        Perform meta-cognitive reflection on the provided error information.

        :param error_info: Information regarding errors in the reasoning path.
        :return: Feedback and suggestions for revising the reasoning path.
        """
        logger.info(
            "Performing meta-reflection on error information."
        )
        prompt = (
            f"Analyze the following error information and suggest modifications to improve the reasoning process:\n"
            f"{error_info}\n"
            "Provide clear and actionable feedback."
        )
        feedback = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=feedback
        )
        logger.debug(f"Meta-reflection feedback: {feedback}")
        return feedback

    def revise_path(self, path: str, feedback: str) -> List[str]:
        """
        Revise the reasoning path based on the provided feedback.

        :param path: The original reasoning path.
        :param feedback: Feedback from meta-cognitive reflection.
        :return: A list of revised reasoning paths.
        """
        logger.info("Revising reasoning path based on feedback.")
        prompt = (
            f"Given the reasoning path:\n'{path}'\n\n"
            f"and the following feedback:\n'{feedback}'\n\n"
            "Generate revised reasoning paths that address the issues raised. "
            "Present each revised path on a new line."
        )
        response = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=response
        )
        revised_paths = [
            line.strip()
            for line in response.split("\n")
            if line.strip()
        ]
        logger.debug(f"Revised paths: {revised_paths}")
        return revised_paths

    def select_promising_paths(self, paths: List[str]) -> List[str]:
        """
        Select the most promising reasoning paths from a list of candidates.

        :param paths: A list of candidate reasoning paths.
        :return: A pruned list containing the most promising paths (max MAX_PATHS_PER_ITERATION).
        """
        if not paths:
            logger.warning("No paths provided for selection")
            return []

        # If already within limit, return as is
        if len(paths) <= MAX_PATHS_PER_ITERATION:
            logger.info(f"Path count ({len(paths)}) within limit, keeping all")
            return paths

        logger.info(f"Selecting top {MAX_PATHS_PER_ITERATION} from {len(paths)} paths")

        # Truncate paths for display to avoid overwhelming the LLM
        paths_display = [p[:200] + "..." if len(p) > 200 else p for p in paths]

        prompt = (
            f"Evaluate the following {len(paths)} reasoning paths and select ONLY the {MAX_PATHS_PER_ITERATION} most promising ones. "
            f"Return EXACTLY {MAX_PATHS_PER_ITERATION} paths, each on a new line. Do not add commentary.\n\n"
            "Paths:\n"
            + "\n".join(f"{i+1}. {p}" for i, p in enumerate(paths_display))
        )
        response = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=response
        )

        selected_paths = [
            line.strip()
            for line in response.split("\n")
            if line.strip() and not line.strip().startswith('#')
        ]

        # Hard limit enforcement - take first MAX_PATHS_PER_ITERATION
        selected_paths = selected_paths[:MAX_PATHS_PER_ITERATION]

        # If LLM failed to return paths, fall back to first N original paths
        if len(selected_paths) < MAX_PATHS_PER_ITERATION:
            logger.warning(f"LLM returned only {len(selected_paths)} paths, using first {MAX_PATHS_PER_ITERATION} original paths")
            selected_paths = paths[:MAX_PATHS_PER_ITERATION]

        logger.info(f"Selected {len(selected_paths)} paths for next iteration")
        return selected_paths

    def synthesize_solution(
        self, paths: List[str], memory_pool: List[str]
    ) -> str:
        """
        Synthesize a final solution from the promising reasoning paths and historical memory.

        :param paths: The current promising reasoning paths.
        :param memory_pool: A list of all previously generated reasoning paths.
        :return: A coherent final solution.
        """
        logger.info(
            "Synthesizing final solution from promising paths."
        )
        prompt = (
            "Based on the following promising reasoning paths:\n"
            f"{chr(10).join(paths)}\n\n"
            "and the historical reasoning memory:\n"
            f"{chr(10).join(memory_pool)}\n\n"
            "Synthesize a final, coherent solution to the problem."
        )
        solution = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=solution
        )
        logger.debug(f"Synthesized solution: {solution}")
        return solution

    def run(self, task: str) -> str:
        """
        Execute the Iterative Reflective Expansion process on the provided problem.

        :param task: The problem statement.
        :return: The final solution generated after iterative reasoning.
        """
        logger.info(
            f"Starting IRE reasoning | Max iterations: {self.max_iterations} | Task: {task[:100]}..."
        )

        candidate_paths = self.generate_initial_hypotheses(task)
        logger.info(f"Generated {len(candidate_paths)} initial hypotheses")

        # Limit initial paths
        if len(candidate_paths) > MAX_PATHS_PER_ITERATION:
            logger.warning(f"Limiting initial paths from {len(candidate_paths)} to {MAX_PATHS_PER_ITERATION}")
            candidate_paths = candidate_paths[:MAX_PATHS_PER_ITERATION]

        memory_pool: List[str] = []
        best_score_overall = 0.0
        early_termination = False

        for iteration in range(self.max_iterations):
            logger.info(
                f"\n{'='*60}\nIteration {iteration + 1}/{self.max_iterations} | Processing {len(candidate_paths)} paths\n{'='*60}"
            )

            expanded_paths: List[str] = []
            iteration_best_score = 0.0
            high_quality_paths = 0

            for idx, path in enumerate(candidate_paths):
                logger.info(f"[Path {idx + 1}/{len(candidate_paths)}] Simulating...")
                outcome, score, error_info = self.simulate_path(path)

                # Track best score
                iteration_best_score = max(iteration_best_score, score)
                best_score_overall = max(best_score_overall, score)

                # Check for early termination
                if score >= EARLY_TERMINATION_SCORE:
                    high_quality_paths += 1
                    logger.info(f"High-quality path found (score: {score:.2f})")
                    expanded_paths.append(path)

                    # Early termination if we have excellent solution
                    if score >= 0.9:
                        logger.info(f"Excellent solution found (score: {score:.2f})! Triggering early termination.")
                        expanded_paths = [path]  # Use only this path
                        early_termination = True
                        break

                elif score < SCORE_THRESHOLD:
                    # Only revise if score is below threshold
                    logger.info(f"Path scored {score:.2f} (below {SCORE_THRESHOLD}), revising...")
                    if error_info:
                        feedback = self.meta_reflect(error_info)
                        revised_paths = self.revise_path(path, feedback)
                        # Limit number of revisions per path
                        revised_paths = revised_paths[:3]
                        expanded_paths.extend(revised_paths)
                        logger.info(f"Generated {len(revised_paths)} revised paths")
                    else:
                        # No explicit errors, keep original path
                        expanded_paths.append(path)
                else:
                    # Good enough, keep it
                    logger.info(f"Path scored {score:.2f}, keeping as-is")
                    expanded_paths.append(path)

            logger.info(
                f"\nIteration {iteration + 1} Summary:\n"
                f"  - Paths processed: {len(candidate_paths)}\n"
                f"  - Expanded to: {len(expanded_paths)} paths\n"
                f"  - Best score this iteration: {iteration_best_score:.2f}\n"
                f"  - Best score overall: {best_score_overall:.2f}\n"
                f"  - High-quality paths: {high_quality_paths}"
            )

            # Check for early termination
            if early_termination:
                logger.info("Early termination triggered - excellent solution found")
                memory_pool.extend(candidate_paths)
                candidate_paths = expanded_paths
                break

            # If we have multiple high-quality paths, we can stop iterating
            if high_quality_paths >= 2 and iteration >= 1:
                logger.info(f"Found {high_quality_paths} high-quality paths, stopping iteration")
                memory_pool.extend(candidate_paths)
                candidate_paths = expanded_paths
                break

            memory_pool.extend(candidate_paths)

            # Select promising paths for next iteration
            candidate_paths = self.select_promising_paths(expanded_paths)

            # Safety check: if no paths remain, break
            if not candidate_paths:
                logger.warning("No candidate paths remain, terminating early")
                candidate_paths = expanded_paths[:MAX_PATHS_PER_ITERATION] if expanded_paths else []
                break

        logger.info(f"\n{'='*60}\nSynthesizing final solution from {len(candidate_paths)} paths\n{'='*60}")
        self.synthesize_solution(candidate_paths, memory_pool)
        logger.info("IRE reasoning complete.")

        return history_output_formatter(
            self.conversation, self.output_type
        )


# def main() -> None:
#     """
#     Main function to execute the Iterative Reflective Expansion algorithm on a sample problem.
#     """
#     problem_statement = "What is the 40th prime number?"
#     reasoning_engine = IterativeReflectiveExpansion(max_iterations=1)
#     final_solution = reasoning_engine.run(problem_statement)
#     print("Final Solution:")
#     print(final_solution)


# if __name__ == "__main__":
#     main()
