"""
Planner-Generator-Evaluator (PGE) Multi-Agent Harness

A domain-agnostic three-agent orchestration harness inspired by the GAN-style
architecture described in Anthropic's harness design research. The harness
coordinates long-running autonomous tasks from a short natural-language prompt,
using an iterative generate-evaluate feedback loop to converge on high-quality
output across any domain.

All three agents communicate through a single shared file on disk.

Flow:
1. User provides a short prompt
2. Planner expands it into a structured plan with evaluation criteria
3. For each step in the plan:
   a. Generator proposes a step contract
   b. Evaluator reviews/amends the contract
   c. Generator executes the step
   d. Evaluator scores output against criteria
   e. If any criterion below threshold: Generator retries with feedback
   f. If all criteria pass: proceed to next step
4. Return final output path and summary

Reference: "Harness design for long-running application development" (Anthropic Engineering, March 2026)
"""

import os
import re
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from swarms.prompts.planner_generator_evaluator_prompts import (
    EVALUATOR_EVALUATE_STEP_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    GENERATOR_EXECUTE_STEP_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    STEP_CONTRACT_NEGOTIATION_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.swarm_id import swarm_id
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="planner_generator_evaluator")


class StepContract:
    """Step contract negotiated between Generator and Evaluator."""

    def __init__(
        self,
        step_number: int,
        title: str = "",
        acceptance_criteria: str = "",
        approved: bool = False,
        amendments: str = "",
    ):
        self.step_number = step_number
        self.title = title
        self.acceptance_criteria = acceptance_criteria
        self.approved = approved
        self.amendments = amendments

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "title": self.title,
            "acceptance_criteria": self.acceptance_criteria,
            "approved": self.approved,
            "amendments": self.amendments,
        }


class EvaluationReport:
    """Per-step evaluation report with criterion scores and pass/fail status."""

    def __init__(
        self,
        step_number: int,
        criterion_scores: Optional[Dict[str, float]] = None,
        criterion_thresholds: Optional[Dict[str, float]] = None,
        passed: bool = False,
        actionable_feedback: str = "",
        summary: str = "",
        raw_evaluation: str = "",
    ):
        self.step_number = step_number
        self.criterion_scores = criterion_scores or {}
        self.criterion_thresholds = criterion_thresholds or {}
        self.passed = passed
        self.actionable_feedback = actionable_feedback
        self.summary = summary
        self.raw_evaluation = raw_evaluation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "criterion_scores": self.criterion_scores,
            "criterion_thresholds": self.criterion_thresholds,
            "passed": self.passed,
            "actionable_feedback": self.actionable_feedback,
            "summary": self.summary,
        }


class HarnessResult:
    """Final output container for a PGE harness run. Access via harness.last_result after run()."""

    def __init__(
        self,
        output_path: str = "",
        plan: str = "",
        step_logs: Optional[List[Dict[str, Any]]] = None,
        total_duration: float = 0.0,
        total_steps_completed: int = 0,
        total_retries: int = 0,
    ):
        self.output_path = output_path
        self.plan = plan
        self.step_logs = step_logs or []
        self.total_duration = total_duration
        self.total_steps_completed = total_steps_completed
        self.total_retries = total_retries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_path": self.output_path,
            "plan": self.plan,
            "step_logs": self.step_logs,
            "total_duration": self.total_duration,
            "total_steps_completed": self.total_steps_completed,
            "total_retries": self.total_retries,
        }


class PlannerGeneratorEvaluator:
    """
    A domain-agnostic three-agent orchestration harness using a Planner,
    Generator, and Evaluator in a GAN-style feedback loop.

    The harness coordinates long-running autonomous tasks from a short
    natural-language prompt. All inter-agent communication flows through
    a single shared file on disk.

    Architecture:
        prompt --> [ Planner ] --> [ Generator ] <-----> [ Evaluator ]
                                        ^                    |
                                        |   feedback loop    |
                                        +--------------------+

    The Planner expands the prompt into a structured plan with evaluation
    criteria. The Generator executes each step, negotiating contracts with
    the Evaluator. The Evaluator scores output and provides actionable
    feedback for iterative refinement.

    Initialization Options:
        1. Use defaults: just provide model_name (creates all three agents internally)
        2. Pass custom agents: provide planner_agent, generator_agent, evaluator_agent
           with your own tools, MCP configs, or any Agent settings

    Args:
        id: Unique identifier for this harness instance.
        name: Human-readable name for this harness.
        description: Description of the harness purpose.
        model_name: Model identifier for all three agents (used when creating defaults).
        planner_model_name: Override model for the Planner agent.
        generator_model_name: Override model for the Generator agent.
        evaluator_model_name: Override model for the Evaluator agent.
        max_steps: Upper bound on plan steps to execute.
        max_retries_per_step: Max evaluation failures before advancing to next step.
        working_directory: Directory where output is produced.
        shared_state_path: Path for the shared state file. Auto-generated if None.
        default_thresholds: Fallback score thresholds by criterion name.
        output_type: Format for the output conversation history.
        verbose: Whether to enable verbose logging.
        planner_agent: Optional pre-configured Agent for planning.
        generator_agent: Optional pre-configured Agent for generation (e.g., with file/code tools).
        evaluator_agent: Optional pre-configured Agent for evaluation (e.g., with Playwright MCP).

    Examples:
        >>> # Simple usage with defaults
        >>> harness = PlannerGeneratorEvaluator(model_name="gpt-4.1")
        >>> result = harness.run("Write a market analysis report for EV batteries")

        >>> # With custom agents that have tools (e.g., Evaluator with Playwright MCP)
        >>> from swarms import Agent
        >>> evaluator = Agent(
        ...     agent_name="PGE-Evaluator",
        ...     model_name="gpt-4.1",
        ...     max_loops=1,
        ...     mcp_config={"url": "http://localhost:3000/playwright"},
        ... )
        >>> harness = PlannerGeneratorEvaluator(
        ...     model_name="gpt-4.1",
        ...     evaluator_agent=evaluator,
        ... )
        >>> result = harness.run("Build a todo app with React frontend")
    """

    def __init__(
        self,
        id: str = None,
        name: str = "PlannerGeneratorEvaluator",
        description: str = "Three-agent Planner-Generator-Evaluator harness with iterative feedback loop",
        model_name: str = "gpt-4.1",
        planner_model_name: Optional[str] = None,
        generator_model_name: Optional[str] = None,
        evaluator_model_name: Optional[str] = None,
        max_steps: int = 10,
        max_retries_per_step: int = 3,
        working_directory: Optional[str] = None,
        shared_state_path: Optional[str] = None,
        default_thresholds: Optional[Dict[str, float]] = None,
        output_type: OutputType = "dict",
        verbose: bool = False,
        planner_system_prompt: str = PLANNER_SYSTEM_PROMPT,
        generator_system_prompt: str = GENERATOR_SYSTEM_PROMPT,
        evaluator_system_prompt: str = EVALUATOR_SYSTEM_PROMPT,
        planner_agent: Optional[Agent] = None,
        generator_agent: Optional[Agent] = None,
        evaluator_agent: Optional[Agent] = None,
        *args,
        **kwargs,
    ):
        self.id = id or swarm_id()
        self.name = name
        self.description = description
        self.model_name = model_name
        self.planner_model_name = planner_model_name or model_name
        self.generator_model_name = generator_model_name or model_name
        self.evaluator_model_name = evaluator_model_name or model_name
        self.max_steps = max_steps
        self.max_retries_per_step = max_retries_per_step
        self.working_directory = working_directory or os.getcwd()
        self.default_thresholds = default_thresholds or {}
        self.output_type = output_type
        self.verbose = verbose
        self.planner_system_prompt = planner_system_prompt
        self.generator_system_prompt = generator_system_prompt
        self.evaluator_system_prompt = evaluator_system_prompt
        # Set up shared state file path
        if shared_state_path:
            self.shared_state_path = shared_state_path
        else:
            self.shared_state_path = os.path.join(
                self.working_directory,
                f"pge_shared_state_{self.id}.md",
            )

        self.reliability_check()

        # Initialize conversation tracker
        self.conversation = Conversation()

        # Use provided agents or create defaults
        self.planner_agent = planner_agent or self._create_planner()
        self.generator_agent = (
            generator_agent or self._create_generator()
        )
        self.evaluator_agent = (
            evaluator_agent or self._create_evaluator()
        )

        self.last_result = None

    def reliability_check(self):
        """Validate harness configuration."""
        if self.max_steps < 1:
            raise ValueError(
                f"max_steps must be >= 1, got {self.max_steps}"
            )
        if self.max_retries_per_step < 0:
            raise ValueError(
                f"max_retries_per_step must be >= 0, got {self.max_retries_per_step}"
            )
        if not self.model_name:
            raise ValueError("model_name must be provided")

        # Ensure working directory exists
        os.makedirs(self.working_directory, exist_ok=True)

        if self.verbose:
            logger.info(
                f"PlannerGeneratorEvaluator initialized: "
                f"max_steps={self.max_steps}, "
                f"max_retries_per_step={self.max_retries_per_step}"
            )

    def _create_planner(self) -> Agent:
        """Create the Planner agent."""
        return Agent(
            agent_name="PGE-Planner",
            agent_description="Strategic planner that expands prompts into detailed execution plans with evaluation criteria",
            system_prompt=self.planner_system_prompt,
            model_name=self.planner_model_name,
            max_loops=1,
            output_type="final",
            verbose=self.verbose,
        )

    def _create_generator(self) -> Agent:
        """Create the Generator agent."""
        return Agent(
            agent_name="PGE-Generator",
            agent_description="Execution agent that produces concrete output following the plan and responding to evaluation feedback",
            system_prompt=self.generator_system_prompt,
            model_name=self.generator_model_name,
            max_loops=1,
            output_type="final",
            verbose=self.verbose,
        )

    def _create_evaluator(self) -> Agent:
        """Create the Evaluator agent."""
        return Agent(
            agent_name="PGE-Evaluator",
            agent_description="Quality evaluator that scores output against criteria and provides actionable feedback",
            system_prompt=self.evaluator_system_prompt,
            model_name=self.evaluator_model_name,
            max_loops=1,
            output_type="final",
            verbose=self.verbose,
        )

    def _read_shared_state(self) -> str:
        """Read the current contents of the shared state file."""
        if os.path.exists(self.shared_state_path):
            with open(self.shared_state_path, "r") as f:
                return f.read()
        return ""

    def _append_to_shared_state(self, section: str, content: str):
        """Append a new section to the shared state file.

        Args:
            section: Section header/label to identify this content.
            content: The content to append.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"\n\n---\n### [{section}] ({timestamp})\n\n{content}\n"
        )

        with open(self.shared_state_path, "a") as f:
            f.write(entry)

    def _initialize_shared_state(self, prompt: str):
        """Initialize the shared state file with the user prompt.

        Args:
            prompt: The user's original prompt.
        """
        header = (
            f"# PGE Harness Shared State\n\n"
            f"**Harness ID**: {self.id}\n"
            f"**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"---\n\n"
            f"## User Prompt\n\n{prompt}\n"
        )

        with open(self.shared_state_path, "w") as f:
            f.write(header)

    def _run_planner(self, prompt: str) -> str:
        """Run the Planner agent to generate the plan and evaluation criteria.

        Args:
            prompt: The user's original prompt.

        Returns:
            The plan text generated by the Planner.
        """
        shared_state = self._read_shared_state()

        planner_task = (
            f"Read the following shared state file and create a comprehensive plan "
            f"for the user's prompt.\n\n"
            f"--- SHARED STATE ---\n{shared_state}\n--- END SHARED STATE ---\n\n"
            f"Generate a detailed plan with steps and evaluation criteria. "
            f"Write your plan in the structured format specified in your instructions."
        )

        plan = self.planner_agent.run(task=planner_task)

        # Append plan to shared state
        self._append_to_shared_state("PLANNER OUTPUT", plan)

        # Track in conversation
        self.conversation.add(
            role=self.planner_agent.agent_name,
            content=plan,
        )

        if self.verbose:
            logger.info("Planner completed plan generation")

        return plan

    def _extract_step_count(self, plan: str) -> int:
        """Extract the number of steps from the plan.

        Looks for step patterns like "Step 1:", "Step 2:", etc.

        Args:
            plan: The plan text to parse.

        Returns:
            Number of steps found, capped at max_steps.
        """
        # Match patterns like "Step 1", "Step 2:", "- Step 1"
        step_matches = re.findall(
            r"(?:^|\n)\s*(?:-\s*)?Step\s+(\d+)", plan, re.IGNORECASE
        )
        if step_matches:
            count = max(int(s) for s in step_matches)
            return min(count, self.max_steps)

        # Fallback: count numbered items
        numbered_matches = re.findall(r"(?:^|\n)\s*(\d+)\.\s+", plan)
        if numbered_matches:
            count = max(int(n) for n in numbered_matches)
            return min(count, self.max_steps)

        # Default to 3 steps if we can't parse
        return min(3, self.max_steps)

    def _extract_thresholds(self, plan: str) -> Dict[str, float]:
        """Extract evaluation criteria thresholds from the plan.

        Parses the evaluation criteria table to find threshold values.

        Args:
            plan: The plan text to parse.

        Returns:
            Map of criterion name to threshold score.
        """
        thresholds = dict(self.default_thresholds)

        # Match table rows: | criterion | weight | description | threshold |
        table_rows = re.findall(
            r"\|\s*([^|]+?)\s*\|\s*(?:high|standard|low|[^|]*?)\s*\|\s*[^|]*?\s*\|\s*(\d+(?:\.\d+)?)\s*\|",
            plan,
            re.IGNORECASE,
        )
        for criterion, threshold in table_rows:
            criterion = criterion.strip().lower()
            if criterion and criterion not in (
                "criterion",
                "criterion name",
                "name",
            ):
                thresholds[criterion] = float(threshold)

        return thresholds

    def _negotiate_contract(self, step_number: int) -> StepContract:
        """Manage the contract negotiation between Generator and Evaluator.

        Args:
            step_number: The current step number.

        Returns:
            The approved StepContract.
        """
        shared_state = self._read_shared_state()

        # Generator proposes a step contract
        generator_task = (
            f"Read the shared state file and propose a step contract for Step {step_number}.\n\n"
            f"--- SHARED STATE ---\n{shared_state}\n--- END SHARED STATE ---\n\n"
            f"Propose a clear step contract with acceptance criteria, "
            f"expected output, and verification method."
        )

        contract_proposal = self.generator_agent.run(
            task=generator_task
        )
        self._append_to_shared_state(
            f"STEP {step_number} CONTRACT PROPOSAL", contract_proposal
        )
        self.conversation.add(
            role=self.generator_agent.agent_name,
            content=f"[Step {step_number} Contract] {contract_proposal}",
        )

        # Evaluator reviews the contract
        updated_state = self._read_shared_state()
        evaluator_task = STEP_CONTRACT_NEGOTIATION_PROMPT.format(
            step_number=step_number
        ) + (
            f"\n\n--- SHARED STATE ---\n{updated_state}\n--- END SHARED STATE ---"
        )

        contract_review = self.evaluator_agent.run(
            task=evaluator_task
        )
        self._append_to_shared_state(
            f"STEP {step_number} CONTRACT REVIEW", contract_review
        )
        self.conversation.add(
            role=self.evaluator_agent.agent_name,
            content=f"[Step {step_number} Contract Review] {contract_review}",
        )

        # Build the StepContract object
        approved = (
            "approved" in contract_review.lower()
            and "amendments required" not in contract_review.lower()
        )

        contract = StepContract(
            step_number=step_number,
            title=f"Step {step_number}",
            acceptance_criteria=contract_proposal,
            approved=approved,
            amendments=contract_review if not approved else "",
        )

        if self.verbose:
            status = (
                "approved" if approved else "amendments requested"
            )
            logger.info(f"Step {step_number} contract {status}")

        return contract

    def _execute_step(
        self,
        step_number: int,
        feedback: str = "",
        retry_count: int = 0,
        score_trajectory: str = "",
    ) -> str:
        """Have the Generator execute a step.

        Args:
            step_number: The current step number.
            feedback: Evaluation feedback from a previous attempt (empty on first try).
            retry_count: Current retry attempt number.
            score_trajectory: Description of how scores changed across retries (refine vs pivot signal).

        Returns:
            The Generator's output for this step.
        """
        shared_state = self._read_shared_state()

        feedback_context = ""
        if feedback:
            feedback_context = (
                f"IMPORTANT: This is retry attempt {retry_count}. "
                f"The Evaluator provided the following feedback on your previous attempt. "
                f"Address each point specifically.\n\n"
                f"{score_trajectory}\n\n"
                f"Evaluator feedback:\n{feedback}"
            )

        generator_task = (
            GENERATOR_EXECUTE_STEP_PROMPT.format(
                step_number=step_number,
                feedback_context=feedback_context,
            )
            + f"\n\n--- SHARED STATE ---\n{shared_state}\n--- END SHARED STATE ---"
        )

        output = self.generator_agent.run(task=generator_task)

        label = f"STEP {step_number} WORK LOG"
        if retry_count > 0:
            label += f" (Retry {retry_count})"
        self._append_to_shared_state(label, output)

        self.conversation.add(
            role=self.generator_agent.agent_name,
            content=f"[Step {step_number} Output (attempt {retry_count + 1})] {output}",
        )

        if self.verbose:
            logger.info(
                f"Generator completed step {step_number} "
                f"(attempt {retry_count + 1})"
            )

        return output

    def _evaluate_step(
        self, step_number: int, thresholds: Dict[str, float]
    ) -> EvaluationReport:
        """Have the Evaluator score the Generator's output.

        Args:
            step_number: The current step number.
            thresholds: Map of criterion name to minimum passing score.

        Returns:
            Structured EvaluationReport with scores and findings.
        """
        shared_state = self._read_shared_state()

        evaluator_task = (
            EVALUATOR_EVALUATE_STEP_PROMPT.format(
                step_number=step_number
            )
            + f"\n\n--- SHARED STATE ---\n{shared_state}\n--- END SHARED STATE ---"
        )

        raw_evaluation = self.evaluator_agent.run(task=evaluator_task)
        self._append_to_shared_state(
            f"STEP {step_number} EVALUATION", raw_evaluation
        )
        self.conversation.add(
            role=self.evaluator_agent.agent_name,
            content=f"[Step {step_number} Evaluation] {raw_evaluation}",
        )

        # Parse scores from the evaluation
        report = self._parse_evaluation(
            step_number, raw_evaluation, thresholds
        )

        if self.verbose:
            status = "PASSED" if report.passed else "FAILED"
            logger.info(
                f"Step {step_number} evaluation: {status} "
                f"(scores: {report.criterion_scores})"
            )

        return report

    def _compute_score_trajectory(
        self, evaluations: List[Dict[str, Any]]
    ) -> str:
        """Compute score trajectory across retries to signal refine vs pivot.

        Args:
            evaluations: List of evaluation dicts from previous attempts.

        Returns:
            A string describing the score trajectory for the Generator.
        """
        if len(evaluations) < 2:
            return ""

        prev = evaluations[-2].get("criterion_scores", {})
        curr = evaluations[-1].get("criterion_scores", {})

        if not prev or not curr:
            return ""

        # Compare average scores
        prev_avg = sum(prev.values()) / len(prev) if prev else 0
        curr_avg = sum(curr.values()) / len(curr) if curr else 0

        if curr_avg > prev_avg:
            return (
                f"SCORE TRAJECTORY: IMPROVING (avg {prev_avg:.1f} -> {curr_avg:.1f}). "
                f"Strategy: REFINE your current approach — fix the specific issues noted below."
            )
        elif curr_avg < prev_avg:
            return (
                f"SCORE TRAJECTORY: DECLINING (avg {prev_avg:.1f} -> {curr_avg:.1f}). "
                f"Strategy: PIVOT to a fundamentally different approach — your current direction is not working."
            )
        else:
            return (
                f"SCORE TRAJECTORY: STAGNANT (avg {curr_avg:.1f}). "
                f"Strategy: PIVOT — try a substantially different approach to break through."
            )

    def _parse_evaluation(
        self,
        step_number: int,
        raw_evaluation: str,
        thresholds: Dict[str, float],
    ) -> EvaluationReport:
        """Parse the Evaluator's raw text output into a structured report.

        Args:
            step_number: The step number being evaluated.
            raw_evaluation: Raw text from the Evaluator.
            thresholds: Map of criterion name to minimum passing score.

        Returns:
            Structured EvaluationReport.
        """
        criterion_scores = {}
        criterion_thresholds = {}

        # Parse table rows: | criterion | score | threshold | status |
        score_rows = re.findall(
            r"\|\s*([^|]+?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(PASS|FAIL)\s*\|",
            raw_evaluation,
            re.IGNORECASE,
        )

        for criterion, score, threshold, status in score_rows:
            criterion = criterion.strip().lower()
            if criterion and criterion not in (
                "criterion",
                "criterion name",
                "name",
            ):
                criterion_scores[criterion] = float(score)
                criterion_thresholds[criterion] = float(threshold)

        # If no table parsed, try alternative patterns:
        # "Criterion: score/10" or "Criterion: score out of 10"
        if not criterion_scores:
            alt_scores = re.findall(
                r"\*?\*?([^*:\n]+?)\*?\*?\s*:\s*(\d+(?:\.\d+)?)\s*(?:/\s*10|out of 10)",
                raw_evaluation,
                re.IGNORECASE,
            )
            for criterion, score in alt_scores:
                criterion = criterion.strip().lower()
                if criterion:
                    criterion_scores[criterion] = float(score)
                    criterion_thresholds[criterion] = thresholds.get(
                        criterion, 6.0
                    )

        # Determine pass/fail
        overall_passed = True
        if criterion_scores:
            for criterion, score in criterion_scores.items():
                threshold = criterion_thresholds.get(
                    criterion,
                    thresholds.get(criterion, 6.0),
                )
                if score < threshold:
                    overall_passed = False
                    break
        else:
            # If we couldn't parse scores, check for explicit PASS/FAIL
            overall_status_match = re.search(
                r"Overall\s+Status\s*:\s*(PASS|FAIL)",
                raw_evaluation,
                re.IGNORECASE,
            )
            if overall_status_match:
                overall_passed = (
                    overall_status_match.group(1).upper() == "PASS"
                )
            else:
                # Default to pass if we truly can't determine
                overall_passed = True

        # Extract actionable feedback section
        feedback_match = re.search(
            r"(?:Actionable Feedback|Improvements?|Recommendations?)\s*:?\s*\n(.*?)(?=\n###|\n---|\Z)",
            raw_evaluation,
            re.IGNORECASE | re.DOTALL,
        )
        actionable_feedback = (
            feedback_match.group(1).strip() if feedback_match else ""
        )

        # Extract summary section
        summary_match = re.search(
            r"(?:Summary|Overall Assessment)\s*:?\s*\n(.*?)(?=\n###|\n---|\Z)",
            raw_evaluation,
            re.IGNORECASE | re.DOTALL,
        )
        summary = (
            summary_match.group(1).strip() if summary_match else ""
        )

        return EvaluationReport(
            step_number=step_number,
            criterion_scores=criterion_scores,
            criterion_thresholds=criterion_thresholds,
            passed=overall_passed,
            actionable_feedback=actionable_feedback,
            summary=summary,
            raw_evaluation=raw_evaluation,
        )

    def run(self, task: str, *args, **kwargs) -> Any:
        """Execute the full PGE harness pipeline.

        Takes a short natural-language prompt and orchestrates the Planner,
        Generator, and Evaluator agents through iterative feedback loops
        to produce high-quality output.

        Args:
            task: A short natural-language description of the desired task.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Formatted conversation history according to output_type.

        Raises:
            ValueError: If task is empty or not a string.
            Exception: If harness execution fails.
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        start_time = time.time()
        step_logs = []
        total_retries = 0

        try:
            # Add task to conversation
            self.conversation.add(role="User", content=task)

            # Initialize shared state file
            self._initialize_shared_state(task)

            if self.verbose:
                logger.info(
                    f"Starting PGE harness for: {task[:100]}..."
                )

            # === Phase 1: Planning ===
            plan = self._run_planner(task)
            step_count = self._extract_step_count(plan)
            thresholds = self._extract_thresholds(plan)

            if self.verbose:
                logger.info(
                    f"Plan generated with {step_count} steps, "
                    f"thresholds: {thresholds}"
                )

            self.conversation.add(
                role="System",
                content=f"--- Planning complete: {step_count} steps identified ---",
            )

            # === Phase 2: Generate-Evaluate Loop ===
            steps_completed = 0

            for step_number in range(1, step_count + 1):
                if self.verbose:
                    logger.info(
                        f"Starting step {step_number}/{step_count}"
                    )

                step_log = {
                    "step_number": step_number,
                    "contract": None,
                    "evaluations": [],
                    "retries": 0,
                    "passed": False,
                }

                # Step 2a-2b: Contract negotiation
                contract = self._negotiate_contract(step_number)
                step_log["contract"] = contract.to_dict()

                # Step 2c-2f: Execute and evaluate loop
                feedback = ""
                retry_count = 0

                while retry_count <= self.max_retries_per_step:
                    # Compute score trajectory for refine-vs-pivot signal
                    trajectory = self._compute_score_trajectory(
                        step_log["evaluations"]
                    )

                    # Generator executes the step
                    self._execute_step(
                        step_number,
                        feedback,
                        retry_count,
                        score_trajectory=trajectory,
                    )

                    # Evaluator reviews the output
                    report = self._evaluate_step(
                        step_number, thresholds
                    )
                    step_log["evaluations"].append(report.to_dict())

                    if report.passed:
                        step_log["passed"] = True
                        steps_completed += 1

                        if self.verbose:
                            logger.info(
                                f"Step {step_number} passed evaluation"
                            )

                        self.conversation.add(
                            role="System",
                            content=f"--- Step {step_number}/{step_count} PASSED ---",
                        )
                        break
                    else:
                        retry_count += 1
                        total_retries += 1
                        step_log["retries"] = retry_count

                        if retry_count > self.max_retries_per_step:
                            logger.warning(
                                f"Step {step_number} failed after "
                                f"{self.max_retries_per_step} retries, advancing"
                            )
                            self.conversation.add(
                                role="System",
                                content=(
                                    f"--- Step {step_number}/{step_count} "
                                    f"FAILED after {self.max_retries_per_step} retries, advancing ---"
                                ),
                            )
                            break

                        # Use evaluation feedback for next attempt
                        feedback = report.raw_evaluation

                        if self.verbose:
                            logger.info(
                                f"Step {step_number} failed evaluation, "
                                f"retry {retry_count}/{self.max_retries_per_step}"
                            )

                step_logs.append(step_log)

            # === Phase 3: Finalize ===
            total_duration = time.time() - start_time

            # Build result (accessible via self.last_result)
            self.last_result = HarnessResult(
                output_path=self.shared_state_path,
                plan=plan,
                step_logs=step_logs,
                total_duration=total_duration,
                total_steps_completed=steps_completed,
                total_retries=total_retries,
            )

            # Add summary to conversation
            summary = (
                f"PGE Harness completed: {steps_completed}/{step_count} steps passed, "
                f"{total_retries} total retries, {total_duration:.1f}s elapsed. "
                f"Shared state: {self.shared_state_path}"
            )
            self.conversation.add(role="System", content=summary)
            self._append_to_shared_state("HARNESS SUMMARY", summary)

            if self.verbose:
                logger.info(summary)

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )

        except Exception as e:
            error_msg = f"PGE Harness execution failed: {str(e)}"
            logger.error(
                f"{error_msg}\n{traceback.format_exc()}\n"
                f"If this persists, report at: https://github.com/kyegomez/swarms/issues"
            )
            raise

    def batched_run(self, tasks: List[str]) -> List[Any]:
        """Run the harness on multiple tasks sequentially.

        Args:
            tasks: List of task prompts to process.

        Returns:
            List of results, one per task.
        """
        return [self.run(task) for task in tasks]

    def get_harness_result(self) -> Dict[str, Any]:
        """Get the current state of the harness as a dictionary.

        Returns:
            Dictionary with harness metadata and conversation history.
        """
        return {
            "id": self.id,
            "name": self.name,
            "shared_state_path": self.shared_state_path,
            "conversation": self.conversation.to_dict(),
        }

    def __call__(self, task: str, *args, **kwargs) -> Any:
        """Make the harness callable."""
        return self.run(task, *args, **kwargs)
