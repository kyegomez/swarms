"""
TalkHier: A hierarchical multi-agent framework for content generation and refinement.
Implements structured communication and evaluation protocols.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from swarms import Agent
from swarms.structs.conversation import Conversation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Defines the possible roles for agents in the system."""

    SUPERVISOR = "supervisor"
    GENERATOR = "generator"
    EVALUATOR = "evaluator"
    REVISOR = "revisor"


@dataclass
class CommunicationEvent:
    """Represents a structured communication event between agents."""

    message: str
    background: Optional[str] = None
    intermediate_output: Optional[Dict[str, Any]] = None
    sender: str = ""
    receiver: str = ""
    timestamp: str = str(datetime.now())


class TalkHier:
    """
    A hierarchical multi-agent system for content generation and refinement.

    Implements the TalkHier framework with structured communication protocols
    and hierarchical refinement processes.

    Attributes:
        max_iterations: Maximum number of refinement iterations
        quality_threshold: Minimum score required for content acceptance
        model_name: Name of the LLM model to use
        base_path: Path for saving agent states
    """

    def __init__(
        self,
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
        model_name: str = "gpt-4",
        base_path: Optional[str] = None,
        return_string: bool = False,
    ):
        """Initialize the TalkHier system."""
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.model_name = model_name
        self.return_string = return_string
        self.base_path = (
            Path(base_path) if base_path else Path("./agent_states")
        )
        self.base_path.mkdir(exist_ok=True)

        # Initialize agents
        self._init_agents()

        # Create conversation
        self.conversation = Conversation()

    def _safely_parse_json(self, json_str: str) -> Dict[str, Any]:
        """
        Safely parse JSON string, handling various formats and potential errors.

        Args:
            json_str: String containing JSON data

        Returns:
            Parsed dictionary
        """
        try:
            # Try direct JSON parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Try extracting JSON from potential text wrapper
                import re

                json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                # Try extracting from markdown code blocks
                code_block_match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```",
                    json_str,
                    re.DOTALL,
                )
                if code_block_match:
                    return json.loads(code_block_match.group(1))
            except Exception as e:
                logger.warning(f"Failed to extract JSON: {str(e)}")

            # Fallback: create structured dict from text
            return {
                "content": json_str,
                "metadata": {
                    "parsed": False,
                    "timestamp": str(datetime.now()),
                },
            }

    def _get_criteria_generator_prompt(self) -> str:
        """Get the prompt for the criteria generator agent."""
        return """You are a Criteria Generator agent responsible for creating task-specific evaluation criteria.
Analyze the task and generate appropriate evaluation criteria based on:
- Task type and complexity
- Required domain knowledge
- Target audience expectations
- Quality requirements

Output all responses in strict JSON format:
{
    "criteria": {
        "criterion_name": {
            "description": "Detailed description of what this criterion measures",
            "importance": "Weight from 0.0-1.0 indicating importance",
            "evaluation_guide": "Guidelines for how to evaluate this criterion"
        }
    },
    "metadata": {
        "task_type": "Classification of the task type",
        "complexity_level": "Assessment of task complexity",
        "domain_focus": "Primary domain or field of the task"
    }
}"""

    def _init_agents(self) -> None:
        """Initialize all agents with their specific roles and prompts."""
        # Main supervisor agent
        self.main_supervisor = Agent(
            agent_name="Main-Supervisor",
            system_prompt=self._get_supervisor_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(
                self.base_path / "main_supervisor.json"
            ),
            verbose=True,
        )

        # Generator agent
        self.generator = Agent(
            agent_name="Content-Generator",
            system_prompt=self._get_generator_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "generator.json"),
            verbose=True,
        )

        # Criteria Generator agent
        self.criteria_generator = Agent(
            agent_name="Criteria-Generator",
            system_prompt=self._get_criteria_generator_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(
                self.base_path / "criteria_generator.json"
            ),
            verbose=True,
        )

        # Evaluators without criteria (will be set during run)
        self.evaluators = []
        for i in range(3):
            self.evaluators.append(
                Agent(
                    agent_name=f"Evaluator-{i}",
                    system_prompt=self._get_evaluator_prompt(i),
                    model_name=self.model_name,
                    max_loops=1,
                    saved_state_path=str(
                        self.base_path / f"evaluator_{i}.json"
                    ),
                    verbose=True,
                )
            )

        # Revisor agent
        self.revisor = Agent(
            agent_name="Content-Revisor",
            system_prompt=self._get_revisor_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "revisor.json"),
            verbose=True,
        )

    def _generate_dynamic_criteria(self, task: str) -> Dict[str, str]:
        """
        Generate dynamic evaluation criteria based on the task.

        Args:
            task: Content generation task description

        Returns:
            Dictionary containing dynamic evaluation criteria
        """
        # Example dynamic criteria generation logic
        if "technical" in task.lower():
            return {
                "accuracy": "Technical correctness and source reliability",
                "clarity": "Readability and logical structure",
                "depth": "Comprehensive coverage of technical details",
                "engagement": "Interest level and relevance to the audience",
                "technical_quality": "Grammar, spelling, and formatting",
            }
        else:
            return {
                "accuracy": "Factual correctness and source reliability",
                "clarity": "Readability and logical structure",
                "coherence": "Logical consistency and argument structure",
                "engagement": "Interest level and relevance to the audience",
                "completeness": "Coverage of the topic and depth",
                "technical_quality": "Grammar, spelling, and formatting",
            }

    def _get_supervisor_prompt(self) -> str:
        """Get the prompt for the supervisor agent."""
        return """You are a Supervisor agent responsible for orchestrating the content generation process and selecting the best evaluation criteria.

You must:
1. Analyze tasks and develop strategies
2. Review multiple evaluator feedback
3. Select the most appropriate evaluation based on:
   - Completeness of criteria
   - Relevance to task
   - Quality of feedback
4. Provide clear instructions for content revision

Output all responses in strict JSON format:
{
    "thoughts": {
        "task_analysis": "Analysis of requirements, audience, scope",
        "strategy": "Step-by-step plan and success metrics",
        "evaluation_selection": {
            "chosen_evaluator": "ID of selected evaluator",
            "reasoning": "Why this evaluation was chosen",
            "key_criteria": ["List of most important criteria"]
        }
    },
    "next_action": {
        "agent": "Next agent to engage",
        "instruction": "Detailed instructions with context"
    }
}"""

    def _get_generator_prompt(self) -> str:
        """Get the prompt for the generator agent."""
        return """You are a Generator agent responsible for creating high-quality, original content. Your role is to produce content that is engaging, informative, and tailored to the target audience.

When generating content:
- Thoroughly research and fact-check all information
- Structure content logically with clear flow
- Use appropriate tone and language for the target audience
- Include relevant examples and explanations
- Ensure content is original and plagiarism-free
- Consider SEO best practices where applicable

Output all responses in strict JSON format:
{
    "content": {
        "main_body": "The complete generated content with proper formatting and structure",
        "metadata": {
            "word_count": "Accurate word count of main body",
            "target_audience": "Detailed audience description",
            "key_points": ["List of main points covered"],
            "sources": ["List of reference sources if applicable"],
            "readability_level": "Estimated reading level",
            "tone": "Description of content tone"
        }
    }
}"""

    def _get_evaluator_prompt(self, evaluator_id: int) -> str:
        """Get the base prompt for an evaluator agent."""
        return f"""You are Evaluator {evaluator_id}, responsible for critically assessing content quality. Your evaluation must be thorough, objective, and constructive.

When receiving content to evaluate:
1. First analyze the task description to determine appropriate evaluation criteria
2. Generate specific criteria based on task requirements
3. Evaluate content against these criteria
4. Provide detailed feedback for each criterion

Output all responses in strict JSON format:
{{
    "generated_criteria": {{
        "criteria_name": "description of what this criterion measures",
        // Add more criteria based on task analysis
    }},
    "scores": {{
        "overall": "0.0-1.0 composite score",
        "categories": {{
            // Scores for each generated criterion
            "criterion_name": "0.0-1.0 score with evidence"
        }}
    }},
    "feedback": [
        "Specific, actionable improvement suggestions per criterion"
    ],
    "strengths": ["Notable positive aspects"],
    "weaknesses": ["Areas needing improvement"]
}}"""

    def _get_revisor_prompt(self) -> str:
        """Get the prompt for the revisor agent."""
        return """You are a Revisor agent responsible for improving content based on evaluator feedback. Your role is to enhance content while maintaining its core message and purpose.

When revising content:
- Address all evaluator feedback systematically
- Maintain consistency in tone and style
- Preserve accurate information
- Enhance clarity and flow
- Fix technical issues
- Optimize for target audience
- Track all changes made

Output all responses in strict JSON format:
{
    "revised_content": {
        "main_body": "Complete revised content incorporating all improvements",
        "metadata": {
            "word_count": "Updated word count",
            "changes_made": [
                "Detailed list of specific changes and improvements",
                "Reasoning for each major revision",
                "Feedback points addressed"
            ],
            "improvement_summary": "Overview of main enhancements",
            "preserved_elements": ["Key elements maintained from original"],
            "revision_approach": "Strategy used for revisions"
        }
    }
}"""

    def _generate_criteria_for_task(
        self, task: str
    ) -> Dict[str, Any]:
        """Generate evaluation criteria for the given task."""
        try:
            criteria_input = {
                "task": task,
                "instruction": "Generate specific evaluation criteria for this task.",
            }

            criteria_response = self.criteria_generator.run(
                json.dumps(criteria_input)
            )
            self.conversation.add(
                role="Criteria-Generator", content=criteria_response
            )

            return self._safely_parse_json(criteria_response)
        except Exception as e:
            logger.error(f"Error generating criteria: {str(e)}")
            return {"criteria": {}}

    def _create_comm_event(
        self, sender: Agent, receiver: Agent, response: Dict
    ) -> CommunicationEvent:
        """Create a structured communication event between agents."""
        return CommunicationEvent(
            message=response.get("message", ""),
            background=response.get("background", ""),
            intermediate_output=response.get(
                "intermediate_output", {}
            ),
            sender=sender.agent_name,
            receiver=receiver.agent_name,
        )

    def _evaluate_content(
        self, content: Union[str, Dict], task: str
    ) -> Dict[str, Any]:
        """Coordinate evaluation process with parallel evaluator execution."""
        try:
            content_dict = (
                self._safely_parse_json(content)
                if isinstance(content, str)
                else content
            )
            criteria_data = self._generate_criteria_for_task(task)

            def run_evaluator(evaluator, eval_input):
                response = evaluator.run(json.dumps(eval_input))
                return {
                    "evaluator_id": evaluator.agent_name,
                    "evaluation": self._safely_parse_json(response),
                }

            eval_inputs = [
                {
                    "task": task,
                    "content": content_dict,
                    "criteria": criteria_data.get("criteria", {}),
                }
                for _ in self.evaluators
            ]

            with ThreadPoolExecutor() as executor:
                evaluations = list(
                    executor.map(
                        lambda x: run_evaluator(*x),
                        zip(self.evaluators, eval_inputs),
                    )
                )

            supervisor_input = {
                "evaluations": evaluations,
                "task": task,
                "instruction": "Synthesize feedback",
            }
            supervisor_response = self.main_supervisor.run(
                json.dumps(supervisor_input)
            )
            aggregated_eval = self._safely_parse_json(
                supervisor_response
            )

            # Track communication
            comm_event = self._create_comm_event(
                self.main_supervisor, self.revisor, aggregated_eval
            )
            self.conversation.add(
                role="Communication",
                content=json.dumps(asdict(comm_event)),
            )

            return aggregated_eval

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            return self._get_fallback_evaluation()

    def _get_fallback_evaluation(self) -> Dict[str, Any]:
        """Get a safe fallback evaluation result."""
        return {
            "scores": {
                "overall": 0.5,
                "categories": {
                    "accuracy": 0.5,
                    "clarity": 0.5,
                    "coherence": 0.5,
                },
            },
            "feedback": ["Evaluation failed"],
            "metadata": {
                "timestamp": str(datetime.now()),
                "is_fallback": True,
            },
        }

    def _aggregate_evaluations(
        self, evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate multiple evaluation results into a single evaluation."""
        try:
            # Collect all unique criteria from evaluators
            all_criteria = set()
            for eval_data in evaluations:
                categories = eval_data.get("scores", {}).get(
                    "categories", {}
                )
                all_criteria.update(categories.keys())

            # Initialize score aggregation
            aggregated_scores = {
                criterion: [] for criterion in all_criteria
            }
            overall_scores = []
            all_feedback = []

            # Collect scores and feedback
            for eval_data in evaluations:
                scores = eval_data.get("scores", {})
                overall_scores.append(scores.get("overall", 0.5))

                categories = scores.get("categories", {})
                for criterion in all_criteria:
                    if criterion in categories:
                        aggregated_scores[criterion].append(
                            categories.get(criterion, 0.5)
                        )

                all_feedback.extend(eval_data.get("feedback", []))

            # Calculate means
            def safe_mean(scores: List[float]) -> float:
                return sum(scores) / len(scores) if scores else 0.5

            return {
                "scores": {
                    "overall": safe_mean(overall_scores),
                    "categories": {
                        criterion: safe_mean(scores)
                        for criterion, scores in aggregated_scores.items()
                    },
                },
                "feedback": list(set(all_feedback)),
                "metadata": {
                    "evaluator_count": len(evaluations),
                    "criteria_used": list(all_criteria),
                    "timestamp": str(datetime.now()),
                },
            }

        except Exception as e:
            logger.error(f"Error in evaluation aggregation: {str(e)}")
            return self._get_fallback_evaluation()

    def _evaluate_and_revise(
        self, content: Union[str, Dict], task: str
    ) -> Dict[str, Any]:
        """Coordinate evaluation and revision process."""
        try:
            # Get evaluations and supervisor selection
            evaluation_result = self._evaluate_content(content, task)

            # Extract selected evaluation and supervisor reasoning
            selected_evaluation = evaluation_result.get(
                "selected_evaluation", {}
            )
            supervisor_reasoning = evaluation_result.get(
                "supervisor_reasoning", {}
            )

            # Prepare revision input with selected evaluation
            revision_input = {
                "content": content,
                "evaluation": selected_evaluation,
                "supervisor_feedback": supervisor_reasoning,
                "instruction": "Revise the content based on the selected evaluation feedback",
            }

            # Get revision from content generator
            revision_response = self.generator.run(
                json.dumps(revision_input)
            )
            revised_content = self._safely_parse_json(
                revision_response
            )

            return {
                "content": revised_content,
                "evaluation": evaluation_result,
            }
        except Exception as e:
            logger.error(f"Evaluation and revision error: {str(e)}")
            return {
                "content": content,
                "evaluation": self._get_fallback_evaluation(),
            }

    def run(self, task: str) -> Dict[str, Any]:
        """
        Generate and iteratively refine content based on the given task.

        Args:
            task: Content generation task description

        Returns:
            Dictionary containing final content and metadata
        """
        logger.info(f"Starting content generation for task: {task}")

        try:
            # Get initial direction from supervisor
            supervisor_response = self.main_supervisor.run(task)

            self.conversation.add(
                role=self.main_supervisor.agent_name,
                content=supervisor_response,
            )

            supervisor_data = self._safely_parse_json(
                supervisor_response
            )

            # Generate initial content
            generator_response = self.generator.run(
                json.dumps(supervisor_data.get("next_action", {}))
            )

            self.conversation.add(
                role=self.generator.agent_name,
                content=generator_response,
            )

            current_content = self._safely_parse_json(
                generator_response
            )

            for iteration in range(self.max_iterations):
                logger.info(f"Starting iteration {iteration + 1}")

                # Evaluate and revise content
                result = self._evaluate_and_revise(
                    current_content, task
                )
                evaluation = result["evaluation"]
                current_content = result["content"]

                # Check if quality threshold is met
                selected_eval = evaluation.get(
                    "selected_evaluation", {}
                )
                overall_score = selected_eval.get("scores", {}).get(
                    "overall", 0.0
                )

                if overall_score >= self.quality_threshold:
                    logger.info(
                        "Quality threshold met, returning content"
                    )
                    return {
                        "content": current_content.get(
                            "content", {}
                        ).get("main_body", ""),
                        "final_score": overall_score,
                        "iterations": iteration + 1,
                        "metadata": {
                            "content_metadata": current_content.get(
                                "content", {}
                            ).get("metadata", {}),
                            "evaluation": evaluation,
                        },
                    }

                # Add to conversation history
                self.conversation.add(
                    role=self.generator.agent_name,
                    content=json.dumps(current_content),
                )

            logger.warning(
                "Max iterations reached without meeting quality threshold"
            )

        except Exception as e:
            logger.error(f"Error in generate_and_refine: {str(e)}")
            current_content = {
                "content": {"main_body": f"Error: {str(e)}"}
            }
            evaluation = self._get_fallback_evaluation()

        if self.return_string:
            return self.conversation.return_history_as_string()
        else:
            return {
                "content": current_content.get("content", {}).get(
                    "main_body", ""
                ),
                "final_score": evaluation["scores"]["overall"],
                "iterations": self.max_iterations,
                "metadata": {
                    "content_metadata": current_content.get(
                        "content", {}
                    ).get("metadata", {}),
                    "evaluation": evaluation,
                    "error": "Max iterations reached",
                },
            }

    def save_state(self) -> None:
        """Save the current state of all agents."""
        for agent in [
            self.main_supervisor,
            self.generator,
            *self.evaluators,
            self.revisor,
        ]:
            try:
                agent.save_state()
            except Exception as e:
                logger.error(
                    f"Error saving state for {agent.agent_name}: {str(e)}"
                )

    def load_state(self) -> None:
        """Load the saved state of all agents."""
        for agent in [
            self.main_supervisor,
            self.generator,
            *self.evaluators,
            self.revisor,
        ]:
            try:
                agent.load_state()
            except Exception as e:
                logger.error(
                    f"Error loading state for {agent.agent_name}: {str(e)}"
                )


# if __name__ == "__main__":
#     try:
#         talkhier = TalkHier(
#             max_iterations=1,
#             quality_threshold=0.8,
#             model_name="gpt-4o",
#             return_string=False,
#         )

#         # Ask for user input
#         task = input("Enter the content generation task description: ")
#         result = talkhier.run(task)

#     except Exception as e:
#         logger.error(f"Error in main execution: {str(e)}")
