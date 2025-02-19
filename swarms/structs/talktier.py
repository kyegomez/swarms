"""
TalkHier: A hierarchical multi-agent framework for content generation and refinement.
Implements structured communication and evaluation protocols.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

        # Evaluators
        self.evaluators = [
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
            for i in range(3)
        ]

        # Revisor agent
        self.revisor = Agent(
            agent_name="Content-Revisor",
            system_prompt=self._get_revisor_prompt(),
            model_name=self.model_name,
            max_loops=1,
            saved_state_path=str(self.base_path / "revisor.json"),
            verbose=True,
        )

    def _get_supervisor_prompt(self) -> str:
        """Get the prompt for the supervisor agent."""
        return """You are a Supervisor agent responsible for orchestrating the content generation process. Your role is to analyze tasks, develop strategies, and coordinate other agents effectively.

You must carefully analyze each task to understand:
- The core objectives and requirements
- Target audience and their needs
- Complexity level and scope
- Any constraints or special considerations

Based on your analysis, develop a clear strategy that:
- Breaks down the task into manageable steps
- Identifies which agents are best suited for each step
- Anticipates potential challenges
- Sets clear success criteria

Output all responses in strict JSON format:
{
    "thoughts": {
        "task_analysis": "Detailed analysis of requirements, audience, scope, and constraints",
        "strategy": "Step-by-step plan including agent allocation and success metrics",
        "concerns": "Potential challenges, edge cases, and mitigation strategies"
    },
    "next_action": {
        "agent": "Specific agent to engage (Generator, Evaluator, or Revisor)",
        "instruction": "Detailed instructions including context, requirements, and expected output"
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
        """Get the prompt for an evaluator agent."""
        return f"""You are Evaluator {evaluator_id}, responsible for critically assessing content quality. Your evaluation must be thorough, objective, and constructive.

Evaluate content across multiple dimensions:
- Accuracy: factual correctness, source reliability
- Clarity: readability, organization, flow
- Coherence: logical consistency, argument structure
- Engagement: interest level, relevance
- Completeness: topic coverage, depth
- Technical quality: grammar, spelling, formatting
- Audience alignment: appropriate level and tone

Output all responses in strict JSON format:
{{
    "scores": {{
        "overall": "0.0-1.0 composite score",
        "categories": {{
            "accuracy": "0.0-1.0 score with evidence",
            "clarity": "0.0-1.0 score with examples",
            "coherence": "0.0-1.0 score with analysis",
            "engagement": "0.0-1.0 score with justification",
            "completeness": "0.0-1.0 score with gaps identified",
            "technical_quality": "0.0-1.0 score with issues noted",
            "audience_alignment": "0.0-1.0 score with reasoning"
        }}
    }},
    "feedback": [
        "Specific, actionable improvement suggestions",
        "Examples of issues found",
        "Recommendations for enhancement"
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

    def _evaluate_content(
        self, content: Union[str, Dict]
    ) -> Dict[str, Any]:
        """
        Coordinate the evaluation of content across multiple evaluators.

        Args:
            content: Content to evaluate (string or dict)

        Returns:
            Combined evaluation results
        """
        try:
            # Ensure content is in correct format
            content_dict = (
                self._safely_parse_json(content)
                if isinstance(content, str)
                else content
            )

            # Collect evaluations
            evaluations = []
            for evaluator in self.evaluators:
                try:
                    eval_response = evaluator.run(
                        json.dumps(content_dict)
                    )

                    self.conversation.add(
                        role=evaluator.agent_name,
                        content=eval_response,
                    )

                    eval_data = self._safely_parse_json(eval_response)
                    evaluations.append(eval_data)
                except Exception as e:
                    logger.warning(f"Evaluator error: {str(e)}")
                    evaluations.append(
                        self._get_fallback_evaluation()
                    )

            # Aggregate results
            return self._aggregate_evaluations(evaluations)

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
        """
        Aggregate multiple evaluation results into a single evaluation.

        Args:
            evaluations: List of evaluation results

        Returns:
            Combined evaluation
        """
        # Calculate average scores
        overall_scores = []
        accuracy_scores = []
        clarity_scores = []
        coherence_scores = []
        all_feedback = []

        for eval_data in evaluations:
            try:
                scores = eval_data.get("scores", {})
                overall_scores.append(scores.get("overall", 0.5))

                categories = scores.get("categories", {})
                accuracy_scores.append(
                    categories.get("accuracy", 0.5)
                )
                clarity_scores.append(categories.get("clarity", 0.5))
                coherence_scores.append(
                    categories.get("coherence", 0.5)
                )

                all_feedback.extend(eval_data.get("feedback", []))
            except Exception as e:
                logger.warning(
                    f"Error aggregating evaluation: {str(e)}"
                )

        def safe_mean(scores: List[float]) -> float:
            return sum(scores) / len(scores) if scores else 0.5

        return {
            "scores": {
                "overall": safe_mean(overall_scores),
                "categories": {
                    "accuracy": safe_mean(accuracy_scores),
                    "clarity": safe_mean(clarity_scores),
                    "coherence": safe_mean(coherence_scores),
                },
            },
            "feedback": list(set(all_feedback)),  # Remove duplicates
            "metadata": {
                "evaluator_count": len(evaluations),
                "timestamp": str(datetime.now()),
            },
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

                # Evaluate current content
                evaluation = self._evaluate_content(current_content)

                # Check if quality threshold is met
                if (
                    evaluation["scores"]["overall"]
                    >= self.quality_threshold
                ):
                    logger.info(
                        "Quality threshold met, returning content"
                    )
                    return {
                        "content": current_content.get(
                            "content", {}
                        ).get("main_body", ""),
                        "final_score": evaluation["scores"][
                            "overall"
                        ],
                        "iterations": iteration + 1,
                        "metadata": {
                            "content_metadata": current_content.get(
                                "content", {}
                            ).get("metadata", {}),
                            "evaluation": evaluation,
                        },
                    }

                # Revise content if needed
                revision_input = {
                    "content": current_content,
                    "evaluation": evaluation,
                }

                revision_response = self.revisor.run(
                    json.dumps(revision_input)
                )
                current_content = self._safely_parse_json(
                    revision_response
                )

                self.conversation.add(
                    role=self.revisor.agent_name,
                    content=revision_response,
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


if __name__ == "__main__":
    # Example usage
    try:
        talkhier = TalkHier(
            max_iterations=1,
            quality_threshold=0.8,
            model_name="gpt-4o",
            return_string=True,
        )

        task = "Write a comprehensive explanation of quantum computing for beginners"
        result = talkhier.run(task)
        print(result)

        # print(f"Final content: {result['content']}")
        # print(f"Quality score: {result['final_score']}")
        # print(f"Iterations: {result['iterations']}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
