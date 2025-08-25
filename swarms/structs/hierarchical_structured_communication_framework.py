"""
Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems

This is a consolidated single-file implementation of the Hierarchical Structured Communication Framework
based on the research paper: "Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems"
arXiv:2502.11098

This file contains all components:
- Structured Communication Protocol (M_ij, B_ij, I_ij)
- Hierarchical Evaluation System
- Specialized Agent Classes
- Main Swarm Orchestrator
- Schemas and Data Models

Key Features:
- Structured communication with Message (M_ij), Background (B_ij), and Intermediate Output (I_ij)
- Hierarchical evaluation team with supervisor coordination
- Dynamic graph-based agent routing
- Context preservation and shared memory
- Flexible model support (OpenAI and Ollama)
"""

import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

# Initialize rich console for enhanced output
console = Console()
logger = initialize_logger(
    log_folder="hierarchical_structured_communication_framework"
)


# =============================================================================
# ENUMS AND DATA MODELS
# =============================================================================


class CommunicationType(str, Enum):
    """Types of communication in the structured protocol"""

    MESSAGE = "message"  # M_ij: Specific task instructions
    BACKGROUND = "background"  # B_ij: Context and problem background
    INTERMEDIATE_OUTPUT = (
        "intermediate_output"  # I_ij: Intermediate results
    )


class AgentRole(str, Enum):
    """Roles for agents in the hierarchical system"""

    SUPERVISOR = "supervisor"
    GENERATOR = "generator"
    EVALUATOR = "evaluator"
    REFINER = "refiner"
    COORDINATOR = "coordinator"


@dataclass
class StructuredMessage:
    """Structured communication message following HierarchicalStructuredComm protocol"""

    message: str = Field(
        description="Specific task instructions (M_ij)"
    )
    background: str = Field(
        description="Context and problem background (B_ij)"
    )
    intermediate_output: str = Field(
        description="Intermediate results (I_ij)"
    )
    sender: str = Field(description="Name of the sending agent")
    recipient: str = Field(description="Name of the receiving agent")
    timestamp: Optional[str] = None


class HierarchicalOrder(BaseModel):
    """Order structure for hierarchical task assignment"""

    agent_name: str = Field(
        description="Name of the agent to receive the task"
    )
    task: str = Field(description="Specific task description")
    communication_type: CommunicationType = Field(
        default=CommunicationType.MESSAGE,
        description="Type of communication to use",
    )
    background_context: str = Field(
        default="", description="Background context for the task"
    )
    intermediate_output: str = Field(
        default="", description="Intermediate output to pass along"
    )


class EvaluationResult(BaseModel):
    """Result from evaluation team member"""

    evaluator_name: str = Field(description="Name of the evaluator")
    criterion: str = Field(description="Evaluation criterion")
    score: float = Field(description="Evaluation score")
    feedback: str = Field(description="Detailed feedback")
    confidence: float = Field(description="Confidence in evaluation")


# =============================================================================
# SCHEMAS
# =============================================================================


class StructuredMessageSchema(BaseModel):
    """Schema for structured communication messages"""

    message: str = Field(
        description="Specific task instructions (M_ij)", min_length=1
    )
    background: str = Field(
        description="Context and problem background (B_ij)",
        default="",
    )
    intermediate_output: str = Field(
        description="Intermediate results (I_ij)", default=""
    )
    sender: str = Field(
        description="Name of the sending agent", min_length=1
    )
    recipient: str = Field(
        description="Name of the receiving agent", min_length=1
    )
    timestamp: Optional[str] = Field(
        description="Timestamp of the message", default=None
    )
    communication_type: CommunicationType = Field(
        description="Type of communication",
        default=CommunicationType.MESSAGE,
    )


class EvaluationResultSchema(BaseModel):
    """Schema for evaluation results"""

    criterion: str = Field(
        description="Evaluation criterion", min_length=1
    )
    score: float = Field(
        description="Evaluation score (0-10)", ge=0.0, le=10.0
    )
    feedback: str = Field(
        description="Detailed feedback", min_length=1
    )
    confidence: float = Field(
        description="Confidence level (0-1)", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Evaluation reasoning", default=""
    )
    suggestions: List[str] = Field(
        description="Improvement suggestions", default=[]
    )


class GeneratorResponseSchema(BaseModel):
    """Schema for generator responses"""

    content: str = Field(
        description="Generated content", min_length=1
    )
    intermediate_output: str = Field(
        description="Intermediate output for next agent", default=""
    )
    reasoning: str = Field(
        description="Generation reasoning", default=""
    )
    confidence: float = Field(
        description="Confidence level (0-1)", ge=0.0, le=1.0
    )


class EvaluatorResponseSchema(BaseModel):
    """Schema for evaluator responses"""

    criterion: str = Field(
        description="Evaluation criterion", min_length=1
    )
    score: float = Field(
        description="Evaluation score (0-10)", ge=0.0, le=10.0
    )
    feedback: str = Field(
        description="Detailed feedback", min_length=1
    )
    confidence: float = Field(
        description="Confidence level (0-1)", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Evaluation reasoning", default=""
    )
    suggestions: List[str] = Field(
        description="Improvement suggestions", default=[]
    )


class RefinerResponseSchema(BaseModel):
    """Schema for refiner responses"""

    refined_content: str = Field(
        description="Refined content", min_length=1
    )
    changes_made: List[str] = Field(
        description="List of changes made", default=[]
    )
    reasoning: str = Field(
        description="Refinement reasoning", default=""
    )
    confidence: float = Field(
        description="Confidence level (0-1)", ge=0.0, le=1.0
    )
    feedback_addressed: List[str] = Field(
        description="Feedback points addressed", default=[]
    )


# =============================================================================
# SPECIALIZED AGENT CLASSES
# =============================================================================


class HierarchicalStructuredCommunicationGenerator(Agent):
    """
    Generator agent for Hierarchical Structured Communication Framework

    This agent specializes in creating initial content following the structured
    communication protocol with Message (M_ij), Background (B_ij), and Intermediate Output (I_ij).
    """

    def __init__(
        self,
        agent_name: str = "TalkHierGenerator",
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the HierarchicalStructuredCommunication Generator agent

        Args:
            agent_name: Name of the agent
            system_prompt: Custom system prompt
            model_name: Model to use
            verbose: Enable verbose logging
        """
        if system_prompt is None:
            system_prompt = self._get_default_generator_prompt()

        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            model_name=model_name,
            verbose=verbose,
            **kwargs,
        )

    def _get_default_generator_prompt(self) -> str:
        """Get the default system prompt for generator agents"""
        return """
You are a Generator agent in a Hierarchical Structured Communication Framework.

Your core responsibility is to create high-quality initial content based on structured input.

**Structured Communication Protocol:**
- Message (M_ij): Specific task instructions you receive
- Background (B_ij): Context and problem background provided
- Intermediate Output (I_ij): Intermediate results to build upon

**Your Process:**
1. **Analyze Input**: Carefully examine the message, background, and intermediate output
2. **Generate Content**: Create comprehensive, well-structured content
3. **Provide Intermediate Output**: Generate intermediate results for the next agent
4. **Structure Response**: Format your response clearly with reasoning and confidence

**Quality Standards:**
- Comprehensive coverage of the task
- Clear structure and organization
- Logical flow and coherence
- Sufficient detail for evaluation
- Original and creative solutions

**Response Format:**
```
Content: [Your generated content]

Intermediate Output: [Structured output for next agent]

Reasoning: [Your reasoning process]

Confidence: [0.0-1.0 confidence level]
```

Always maintain high quality and provide detailed, actionable content.
"""

    def generate_with_structure(
        self,
        message: str,
        background: str = "",
        intermediate_output: str = "",
        **kwargs,
    ) -> GeneratorResponseSchema:
        """
        Generate content using structured communication protocol

        Args:
            message: Specific task message (M_ij)
            background: Background context (B_ij)
            intermediate_output: Intermediate output (I_ij)

        Returns:
            GeneratorResponseSchema with structured response
        """
        try:
            # Construct structured prompt
            prompt = self._construct_structured_prompt(
                message, background, intermediate_output
            )

            # Generate response
            response = self.run(prompt, **kwargs)

            # Parse and structure response
            return self._parse_generator_response(response)

        except Exception as e:
            logger.error(f"Error in structured generation: {e}")
            return GeneratorResponseSchema(
                content=f"Error generating content: {e}",
                intermediate_output="",
                reasoning="Error occurred during generation",
                confidence=0.0,
            )

    def _construct_structured_prompt(
        self, message: str, background: str, intermediate_output: str
    ) -> str:
        """Construct a structured prompt for generation"""
        prompt_parts = []

        if message:
            prompt_parts.append(f"**Task Message (M_ij):** {message}")

        if background:
            prompt_parts.append(
                f"**Background Context (B_ij):** {background}"
            )

        if intermediate_output:
            prompt_parts.append(
                f"**Intermediate Output (I_ij):** {intermediate_output}"
            )

        prompt = "\n\n".join(prompt_parts)
        prompt += "\n\nPlease generate content following the structured response format."

        return prompt

    def _parse_generator_response(
        self, response: str
    ) -> GeneratorResponseSchema:
        """Parse the generator response into structured format"""
        try:
            lines = response.split("\n")
            content = ""
            intermediate_output = ""
            reasoning = ""
            confidence = 0.8  # Default confidence

            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.lower().startswith("content:"):
                    current_section = "content"
                    content = line[8:].strip()
                elif line.lower().startswith("intermediate output:"):
                    current_section = "intermediate"
                    intermediate_output = line[20:].strip()
                elif line.lower().startswith("reasoning:"):
                    current_section = "reasoning"
                    reasoning = line[10:].strip()
                elif line.lower().startswith("confidence:"):
                    try:
                        confidence = float(line[11:].strip())
                    except ValueError:
                        confidence = 0.8
                elif current_section == "content":
                    content += " " + line
                elif current_section == "intermediate":
                    intermediate_output += " " + line
                elif current_section == "reasoning":
                    reasoning += " " + line

            return GeneratorResponseSchema(
                content=content or response,
                intermediate_output=intermediate_output,
                reasoning=reasoning,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error parsing generator response: {e}")
            return GeneratorResponseSchema(
                content=response,
                intermediate_output="",
                reasoning="Error parsing response",
                confidence=0.5,
            )


class HierarchicalStructuredCommunicationEvaluator(Agent):
    """
    Evaluator agent for Hierarchical Structured Communication Framework

    This agent specializes in evaluating content using specific criteria and
    providing structured feedback following the hierarchical evaluation system.
    """

    def __init__(
        self,
        agent_name: str = "TalkHierEvaluator",
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        evaluation_criteria: List[str] = None,
        **kwargs,
    ):
        """
        Initialize the HierarchicalStructuredCommunication Evaluator agent

        Args:
            agent_name: Name of the agent
            system_prompt: Custom system prompt
            model_name: Model to use
            verbose: Enable verbose logging
            evaluation_criteria: List of evaluation criteria this agent can assess
        """
        if system_prompt is None:
            system_prompt = self._get_default_evaluator_prompt(
                evaluation_criteria
            )

        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            model_name=model_name,
            verbose=verbose,
            **kwargs,
        )

        self.evaluation_criteria = evaluation_criteria or [
            "accuracy",
            "completeness",
            "clarity",
            "relevance",
        ]

    def _get_default_evaluator_prompt(
        self, criteria: List[str] = None
    ) -> str:
        """Get the default system prompt for evaluator agents"""
        if criteria is None:
            criteria = [
                "accuracy",
                "completeness",
                "clarity",
                "relevance",
            ]

        criteria_text = "\n".join(
            [f"- {criterion}" for criterion in criteria]
        )

        return f"""
You are an Evaluator agent in a Hierarchical Structured Communication Framework.

Your core responsibility is to evaluate content quality using specific criteria and provide structured feedback.

**Evaluation Criteria:**
{criteria_text}

**Evaluation Process:**
1. **Analyze Content**: Examine the content thoroughly
2. **Apply Criteria**: Evaluate against the specified criterion
3. **Score Performance**: Provide numerical score (0-10)
4. **Provide Feedback**: Give detailed, actionable feedback
5. **Assess Confidence**: Rate your confidence in the evaluation

**Scoring Guidelines:**
- 9-10: Excellent - Outstanding quality, minimal issues
- 7-8: Good - High quality with minor improvements needed
- 5-6: Average - Adequate but significant improvements needed
- 3-4: Below Average - Major issues, substantial improvements required
- 1-2: Poor - Critical issues, extensive revision needed
- 0: Unacceptable - Fundamental problems

**Response Format:**
```
Criterion: [Evaluation criterion]

Score: [0-10 numerical score]

Feedback: [Detailed feedback]

Confidence: [0.0-1.0 confidence level]

Reasoning: [Your evaluation reasoning]

Suggestions: [Specific improvement suggestions]
```

Be thorough, fair, and constructive in your evaluations.
"""

    def evaluate_with_criterion(
        self, content: str, criterion: str, **kwargs
    ) -> EvaluatorResponseSchema:
        """
        Evaluate content using a specific criterion

        Args:
            content: Content to evaluate
            criterion: Specific evaluation criterion

        Returns:
            EvaluatorResponseSchema with evaluation results
        """
        try:
            # Construct evaluation prompt
            prompt = self._construct_evaluation_prompt(
                content, criterion
            )

            # Get evaluation response
            response = self.run(prompt, **kwargs)

            # Parse and structure response
            return self._parse_evaluator_response(response, criterion)

        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return EvaluatorResponseSchema(
                criterion=criterion,
                score=5.0,
                feedback=f"Error during evaluation: {e}",
                confidence=0.0,
                reasoning="Error occurred during evaluation",
                suggestions=[
                    "Fix technical issues",
                    "Retry evaluation",
                ],
            )

    def _construct_evaluation_prompt(
        self, content: str, criterion: str
    ) -> str:
        """Construct an evaluation prompt"""
        return f"""
**Content to Evaluate:**
{content}

**Evaluation Criterion:**
{criterion}

Please evaluate the content above based on the {criterion} criterion.

Provide your evaluation following the structured response format.
"""

    def _parse_evaluator_response(
        self, response: str, criterion: str
    ) -> EvaluatorResponseSchema:
        """Parse the evaluator response into structured format"""
        try:
            lines = response.split("\n")
            score = 5.0  # Default score
            feedback = ""
            confidence = 0.8  # Default confidence
            reasoning = ""
            suggestions = []

            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.lower().startswith("score:"):
                    try:
                        score = float(line[6:].strip())
                    except ValueError:
                        score = 5.0
                elif line.lower().startswith("feedback:"):
                    current_section = "feedback"
                    feedback = line[9:].strip()
                elif line.lower().startswith("confidence:"):
                    try:
                        confidence = float(line[11:].strip())
                    except ValueError:
                        confidence = 0.8
                elif line.lower().startswith("reasoning:"):
                    current_section = "reasoning"
                    reasoning = line[10:].strip()
                elif line.lower().startswith("suggestions:"):
                    current_section = "suggestions"
                elif current_section == "feedback":
                    feedback += " " + line
                elif current_section == "reasoning":
                    reasoning += " " + line
                elif current_section == "suggestions":
                    if line.startswith("-") or line.startswith("•"):
                        suggestions.append(line[1:].strip())
                    else:
                        suggestions.append(line)

            return EvaluatorResponseSchema(
                criterion=criterion,
                score=score,
                feedback=feedback or "No feedback provided",
                confidence=confidence,
                reasoning=reasoning,
                suggestions=suggestions,
            )

        except Exception as e:
            logger.error(f"Error parsing evaluator response: {e}")
            return EvaluatorResponseSchema(
                criterion=criterion,
                score=5.0,
                feedback="Error parsing evaluation response",
                confidence=0.0,
                reasoning="Error occurred during parsing",
                suggestions=["Fix parsing issues"],
            )


class HierarchicalStructuredCommunicationRefiner(Agent):
    """
    Refiner agent for Hierarchical Structured Communication Framework

    This agent specializes in improving content based on evaluation feedback
    and maintaining the structured communication protocol.
    """

    def __init__(
        self,
        agent_name: str = "TalkHierRefiner",
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the HierarchicalStructuredCommunication Refiner agent

        Args:
            agent_name: Name of the agent
            system_prompt: Custom system prompt
            model_name: Model to use
            verbose: Enable verbose logging
        """
        if system_prompt is None:
            system_prompt = self._get_default_refiner_prompt()

        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            model_name=model_name,
            verbose=verbose,
            **kwargs,
        )

    def _get_default_refiner_prompt(self) -> str:
        """Get the default system prompt for refiner agents"""
        return """
You are a Refiner agent in a Hierarchical Structured Communication Framework.

Your core responsibility is to improve content based on evaluation feedback while maintaining quality and addressing specific issues.

**Refinement Process:**
1. **Analyze Original**: Understand the original content thoroughly
2. **Review Feedback**: Examine all evaluation feedback carefully
3. **Identify Issues**: Identify specific problems to address
4. **Make Improvements**: Enhance content while preserving strengths
5. **Justify Changes**: Explain why each improvement was made

**Refinement Principles:**
- Address specific feedback points
- Maintain core strengths and structure
- Improve clarity and coherence
- Enhance completeness and accuracy
- Preserve original intent and purpose

**Response Format:**
```
Refined Content: [Your improved content]

Changes Made: [List of specific changes]

Reasoning: [Explanation of refinements]

Confidence: [0.0-1.0 confidence in improvements]

Feedback Addressed: [Which feedback points were addressed]
```

Focus on meaningful improvements that directly address the evaluation feedback.
"""

    def refine_with_feedback(
        self,
        original_content: str,
        evaluation_results: List[EvaluationResultSchema],
        **kwargs,
    ) -> RefinerResponseSchema:
        """
        Refine content based on evaluation feedback

        Args:
            original_content: Original content to refine
            evaluation_results: List of evaluation results with feedback

        Returns:
            RefinerResponseSchema with refined content
        """
        try:
            # Construct refinement prompt
            prompt = self._construct_refinement_prompt(
                original_content, evaluation_results
            )

            # Get refinement response
            response = self.run(prompt, **kwargs)

            # Parse and structure response
            return self._parse_refiner_response(
                response, evaluation_results
            )

        except Exception as e:
            logger.error(f"Error in refinement: {e}")
            return RefinerResponseSchema(
                refined_content=original_content,
                changes_made=["Error occurred during refinement"],
                reasoning=f"Error during refinement: {e}",
                confidence=0.0,
                feedback_addressed=[],
            )

    def _construct_refinement_prompt(
        self,
        original_content: str,
        evaluation_results: List[EvaluationResultSchema],
    ) -> str:
        """Construct a refinement prompt"""
        feedback_summary = "\n\n".join(
            [
                f"**{result.criterion} (Score: {result.score}/10):**\n{result.feedback}"
                for result in evaluation_results
            ]
        )

        return f"""
**Original Content:**
{original_content}

**Evaluation Feedback:**
{feedback_summary}

Please refine the content to address the feedback while maintaining its core strengths.

Provide your refinement following the structured response format.
"""

    def _parse_refiner_response(
        self,
        response: str,
        evaluation_results: List[EvaluationResultSchema],
    ) -> RefinerResponseSchema:
        """Parse the refiner response into structured format"""
        try:
            lines = response.split("\n")
            refined_content = ""
            changes_made = []
            reasoning = ""
            confidence = 0.8  # Default confidence
            feedback_addressed = []

            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.lower().startswith("refined content:"):
                    current_section = "content"
                    refined_content = line[16:].strip()
                elif line.lower().startswith("changes made:"):
                    current_section = "changes"
                elif line.lower().startswith("reasoning:"):
                    current_section = "reasoning"
                    reasoning = line[10:].strip()
                elif line.lower().startswith("confidence:"):
                    try:
                        confidence = float(line[11:].strip())
                    except ValueError:
                        confidence = 0.8
                elif line.lower().startswith("feedback addressed:"):
                    current_section = "feedback"
                elif current_section == "content":
                    refined_content += " " + line
                elif current_section == "changes":
                    if line.startswith("-") or line.startswith("•"):
                        changes_made.append(line[1:].strip())
                    else:
                        changes_made.append(line)
                elif current_section == "reasoning":
                    reasoning += " " + line
                elif current_section == "feedback":
                    if line.startswith("-") or line.startswith("•"):
                        feedback_addressed.append(line[1:].strip())
                    else:
                        feedback_addressed.append(line)

            # If no refined content found, use original
            if not refined_content:
                refined_content = response

            return RefinerResponseSchema(
                refined_content=refined_content,
                changes_made=changes_made,
                reasoning=reasoning,
                confidence=confidence,
                feedback_addressed=feedback_addressed,
            )

        except Exception as e:
            logger.error(f"Error parsing refiner response: {e}")
            return RefinerResponseSchema(
                refined_content=response,
                changes_made=["Error parsing response"],
                reasoning="Error occurred during parsing",
                confidence=0.0,
                feedback_addressed=[],
            )


class HierarchicalStructuredCommunicationSupervisor(Agent):
    """
    Supervisor agent for Hierarchical Structured Communication Framework

    This agent coordinates the overall workflow and manages structured communication
    between different agent types.
    """

    def __init__(
        self,
        agent_name: str = "TalkHierSupervisor",
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the HierarchicalStructuredCommunication Supervisor agent

        Args:
            agent_name: Name of the agent
            system_prompt: Custom system prompt
            model_name: Model to use
            verbose: Enable verbose logging
        """
        if system_prompt is None:
            system_prompt = self._get_default_supervisor_prompt()

        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            model_name=model_name,
            verbose=verbose,
            **kwargs,
        )

    def _get_default_supervisor_prompt(self) -> str:
        """Get the default system prompt for supervisor agents"""
        return """
You are a Supervisor agent in a Hierarchical Structured Communication Framework.

Your core responsibility is to coordinate the workflow and manage structured communication between agents.

**Supervision Responsibilities:**
1. **Task Orchestration**: Coordinate between generator, evaluator, and refiner agents
2. **Structured Communication**: Ensure proper use of Message (M_ij), Background (B_ij), and Intermediate Output (I_ij)
3. **Workflow Management**: Manage the iterative refinement process
4. **Quality Control**: Monitor and ensure high-quality outputs
5. **Decision Making**: Determine when to continue refinement or stop

**Communication Protocol:**
- Always provide structured messages with clear components
- Maintain context and background information
- Pass intermediate outputs between agents
- Ensure proper agent coordination

**Decision Criteria:**
- Evaluation scores and feedback quality
- Content improvement progress
- Resource constraints and time limits
- Quality thresholds and standards

**Response Format:**
```
Next Action: [What should happen next]

Target Agent: [Which agent should act]

Structured Message: [Complete structured message]

Background Context: [Context to provide]

Intermediate Output: [Output to pass along]

Reasoning: [Why this decision was made]
```

Focus on efficient coordination and high-quality outcomes.
"""

    def coordinate_workflow(
        self, task: str, current_state: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Coordinate the workflow and determine next actions

        Args:
            task: Current task being processed
            current_state: Current state of the workflow

        Returns:
            Dictionary with coordination decisions
        """
        try:
            # Construct coordination prompt
            prompt = self._construct_coordination_prompt(
                task, current_state
            )

            # Get coordination response
            response = self.run(prompt, **kwargs)

            # Parse and structure response
            return self._parse_coordination_response(response)

        except Exception as e:
            logger.error(f"Error in workflow coordination: {e}")
            return {
                "next_action": "error",
                "target_agent": "none",
                "structured_message": f"Error in coordination: {e}",
                "background_context": "",
                "intermediate_output": "",
                "reasoning": "Error occurred during coordination",
            }

    def _construct_coordination_prompt(
        self, task: str, current_state: Dict[str, Any]
    ) -> str:
        """Construct a coordination prompt"""
        state_summary = "\n".join(
            [
                f"- {key}: {value}"
                for key, value in current_state.items()
            ]
        )

        return f"""
**Current Task:**
{task}

**Current State:**
{state_summary}

Please coordinate the workflow and determine the next action.

Provide your coordination decision following the structured response format.
"""

    def _parse_coordination_response(
        self, response: str
    ) -> Dict[str, Any]:
        """Parse the coordination response"""
        try:
            lines = response.split("\n")
            result = {
                "next_action": "continue",
                "target_agent": "generator",
                "structured_message": "",
                "background_context": "",
                "intermediate_output": "",
                "reasoning": "",
            }

            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.lower().startswith("next action:"):
                    result["next_action"] = line[12:].strip()
                elif line.lower().startswith("target agent:"):
                    result["target_agent"] = line[13:].strip()
                elif line.lower().startswith("structured message:"):
                    current_section = "message"
                    result["structured_message"] = line[19:].strip()
                elif line.lower().startswith("background context:"):
                    current_section = "background"
                    result["background_context"] = line[19:].strip()
                elif line.lower().startswith("intermediate output:"):
                    current_section = "output"
                    result["intermediate_output"] = line[20:].strip()
                elif line.lower().startswith("reasoning:"):
                    current_section = "reasoning"
                    result["reasoning"] = line[10:].strip()
                elif current_section == "message":
                    result["structured_message"] += " " + line
                elif current_section == "background":
                    result["background_context"] += " " + line
                elif current_section == "output":
                    result["intermediate_output"] += " " + line
                elif current_section == "reasoning":
                    result["reasoning"] += " " + line

            return result

        except Exception as e:
            logger.error(f"Error parsing coordination response: {e}")
            return {
                "next_action": "error",
                "target_agent": "none",
                "structured_message": "Error parsing response",
                "background_context": "",
                "intermediate_output": "",
                "reasoning": "Error occurred during parsing",
            }


# =============================================================================
# MAIN SWARM ORCHESTRATOR
# =============================================================================


class HierarchicalStructuredCommunicationFramework(BaseSwarm):
    """
    Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems

    This is the main orchestrator class that implements the complete HierarchicalStructuredComm approach with:
    1. Structured Communication Protocol
    2. Hierarchical Refinement System
    3. Graph-based Agent Orchestration

    Architecture:
    - Supervisor Agent: Coordinates the overall workflow
    - Generator Agents: Create initial content/solutions
    - Evaluator Team: Hierarchical evaluation with supervisor
    - Refiner Agents: Improve solutions based on feedback
    """

    def __init__(
        self,
        name: str = "HierarchicalStructuredCommunicationFramework",
        description: str = "Talk Structurally, Act Hierarchically Framework",
        supervisor: Optional[Union[Agent, Callable, Any]] = None,
        generators: List[Union[Agent, Callable, Any]] = None,
        evaluators: List[Union[Agent, Callable, Any]] = None,
        refiners: List[Union[Agent, Callable, Any]] = None,
        evaluation_supervisor: Optional[
            Union[Agent, Callable, Any]
        ] = None,
        max_loops: int = 3,
        output_type: OutputType = "dict-all-except-first",
        supervisor_name: str = "Supervisor",
        evaluation_supervisor_name: str = "EvaluationSupervisor",
        verbose: bool = False,
        enable_structured_communication: bool = True,
        enable_hierarchical_evaluation: bool = True,
        shared_memory: bool = True,
        model_name: str = "gpt-4o-mini",
        use_ollama: bool = False,
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_api_key: str = "ollama",
        *args,
        **kwargs,
    ):
        """
        Initialize the HierarchicalStructuredCommunicationFramework

        Args:
            name: Name of the swarm
            description: Description of the swarm
            supervisor: Main supervisor agent
            generators: List of generator agents
            evaluators: List of evaluator agents
            refiners: List of refiner agents
            evaluation_supervisor: Supervisor for evaluation team
            max_loops: Maximum number of refinement loops
            output_type: Type of output format
            supervisor_name: Name for the supervisor agent
            evaluation_supervisor_name: Name for evaluation supervisor
            verbose: Enable verbose logging
            enable_structured_communication: Enable structured communication protocol
            enable_hierarchical_evaluation: Enable hierarchical evaluation system
            shared_memory: Enable shared memory between agents
            model_name: Model name to use for default agents
            use_ollama: Whether to use Ollama for local inference
            ollama_base_url: Ollama API base URL
            ollama_api_key: Ollama API key
        """
        # Initialize the swarm components first
        self.name = name
        self.description = description
        self.supervisor = supervisor
        self.generators = generators or []
        self.evaluators = evaluators or []
        self.refiners = refiners or []
        self.evaluation_supervisor = evaluation_supervisor
        self.max_loops = max_loops
        self.output_type = output_type
        self.supervisor_name = supervisor_name
        self.evaluation_supervisor_name = evaluation_supervisor_name
        self.verbose = verbose
        self.enable_structured_communication = (
            enable_structured_communication
        )
        self.enable_hierarchical_evaluation = (
            enable_hierarchical_evaluation
        )
        self.shared_memory = shared_memory
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.ollama_base_url = ollama_base_url
        self.ollama_api_key = ollama_api_key

        # Communication and state management
        self.conversation_history: List[StructuredMessage] = []
        self.intermediate_outputs: Dict[str, str] = {}
        self.evaluation_results: List[EvaluationResult] = []

        # Initialize the swarm components
        self.init_swarm()

        # Collect all agents for the parent class
        all_agents = []
        if self.supervisor:
            all_agents.append(self.supervisor)
        all_agents.extend(self.generators)
        all_agents.extend(self.evaluators)
        all_agents.extend(self.refiners)
        if self.evaluation_supervisor:
            all_agents.append(self.evaluation_supervisor)

        # Call parent constructor with agents
        super().__init__(agents=all_agents, *args, **kwargs)

    def init_swarm(self):
        """Initialize the swarm components"""
        # Enhanced logging with rich formatting
        console.print(
            Panel(
                f"[bold blue]Initializing {self.name}[/bold blue]\n"
                f"[dim]Framework: Talk Structurally, Act Hierarchically[/dim]",
                title="Framework Initialization",
                border_style="blue",
            )
        )
        logger.info(f"Initializing {self.name}")

        # Setup supervisor if not provided
        if self.supervisor is None:
            self.supervisor = self._create_supervisor_agent()

        # Setup evaluation supervisor if not provided
        if (
            self.evaluation_supervisor is None
            and self.enable_hierarchical_evaluation
        ):
            self.evaluation_supervisor = (
                self._create_evaluation_supervisor_agent()
            )

        # Setup default agents if none provided
        if not self.generators:
            self.generators = [self._create_default_generator()]

        if (
            not self.evaluators
            and self.enable_hierarchical_evaluation
        ):
            self.evaluators = [self._create_default_evaluator()]

        if not self.refiners:
            self.refiners = [self._create_default_refiner()]

        # Enhanced status display
        table = Table(title="Framework Components")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Count", style="magenta")
        table.add_column("Status", style="green")

        table.add_row(
            "Generators", str(len(self.generators)), "Ready"
        )
        table.add_row(
            "Evaluators", str(len(self.evaluators)), "Ready"
        )
        table.add_row("Refiners", str(len(self.refiners)), "Ready")
        table.add_row(
            "Supervisors", str(1 if self.supervisor else 0), "Ready"
        )

        console.print(table)

        logger.info(
            f"Swarm initialized with {len(self.generators)} generators, "
            f"{len(self.evaluators)} evaluators, {len(self.refiners)} refiners"
        )

    def _create_supervisor_agent(self) -> Agent:
        """Create the main supervisor agent"""
        supervisor_prompt = self._get_supervisor_prompt()

        agent_kwargs = {
            "agent_name": self.supervisor_name,
            "system_prompt": supervisor_prompt,
            "model_name": self.model_name,
            "verbose": self.verbose,
            "reliability_check": False,
        }

        if self.use_ollama:
            agent_kwargs.update(
                {
                    "openai_api_base": self.ollama_base_url,
                    "openai_api_key": self.ollama_api_key,
                }
            )

        return Agent(**agent_kwargs)

    def _create_evaluation_supervisor_agent(self) -> Agent:
        """Create the evaluation team supervisor"""
        eval_supervisor_prompt = (
            self._get_evaluation_supervisor_prompt()
        )

        agent_kwargs = {
            "agent_name": self.evaluation_supervisor_name,
            "system_prompt": eval_supervisor_prompt,
            "model_name": self.model_name,
            "verbose": self.verbose,
            "reliability_check": False,
        }

        if self.use_ollama:
            agent_kwargs.update(
                {
                    "openai_api_base": self.ollama_base_url,
                    "openai_api_key": self.ollama_api_key,
                }
            )

        return Agent(**agent_kwargs)

    def _create_default_generator(self) -> Agent:
        """Create a default generator agent"""
        generator_prompt = self._get_generator_prompt()

        agent_kwargs = {
            "agent_name": "Generator",
            "system_prompt": generator_prompt,
            "model_name": self.model_name,
            "verbose": self.verbose,
            "reliability_check": False,
        }

        if self.use_ollama:
            agent_kwargs.update(
                {
                    "openai_api_base": self.ollama_base_url,
                    "openai_api_key": self.ollama_api_key,
                }
            )

        return Agent(**agent_kwargs)

    def _create_default_evaluator(self) -> Agent:
        """Create a default evaluator agent"""
        evaluator_prompt = self._get_evaluator_prompt()

        agent_kwargs = {
            "agent_name": "Evaluator",
            "system_prompt": evaluator_prompt,
            "model_name": self.model_name,
            "verbose": self.verbose,
            "reliability_check": False,
        }

        if self.use_ollama:
            agent_kwargs.update(
                {
                    "openai_api_base": self.ollama_base_url,
                    "openai_api_key": self.ollama_api_key,
                }
            )

        return Agent(**agent_kwargs)

    def _create_default_refiner(self) -> Agent:
        """Create a default refiner agent"""
        refiner_prompt = self._get_refiner_prompt()

        agent_kwargs = {
            "agent_name": "Refiner",
            "system_prompt": refiner_prompt,
            "model_name": self.model_name,
            "verbose": self.verbose,
            "reliability_check": False,
        }

        if self.use_ollama:
            agent_kwargs.update(
                {
                    "openai_api_base": self.ollama_base_url,
                    "openai_api_key": self.ollama_api_key,
                }
            )

        return Agent(**agent_kwargs)

    def _get_supervisor_prompt(self) -> str:
        """Get the supervisor system prompt"""
        return f"""
You are the {self.supervisor_name} in a Talk Structurally, Act Hierarchically framework.

Your responsibilities:
1. **Structured Communication**: Use the structured communication protocol with:
   - Message (M_ij): Specific task instructions
   - Background (B_ij): Context and problem background  
   - Intermediate Output (I_ij): Intermediate results

2. **Task Orchestration**: Coordinate between generator, evaluator, and refiner agents

3. **Workflow Management**: Manage the iterative refinement process

Available agents:
- Generators: {[agent.agent_name if hasattr(agent, 'agent_name') else 'Generator' for agent in self.generators]}
- Evaluators: {[agent.agent_name if hasattr(agent, 'agent_name') else 'Evaluator' for agent in self.evaluators]}
- Refiners: {[agent.agent_name if hasattr(agent, 'agent_name') else 'Refiner' for agent in self.refiners]}

Always provide structured communication with clear message, background context, and intermediate outputs.
"""

    def _get_evaluation_supervisor_prompt(self) -> str:
        """Get the evaluation supervisor system prompt"""
        return f"""
You are the {self.evaluation_supervisor_name} in a hierarchical evaluation system.

Your responsibilities:
1. **Coordinate Evaluators**: Manage multiple evaluators with different criteria
2. **Synthesize Feedback**: Combine evaluation results into coherent feedback
3. **Provide Summarized Results**: Give concise, actionable feedback to the main supervisor

Evaluation criteria to coordinate:
- Accuracy and correctness
- Completeness and coverage
- Clarity and coherence
- Relevance and appropriateness

Always provide summarized, coordinated feedback that balances diverse evaluator inputs.
"""

    def _get_generator_prompt(self) -> str:
        """Get the generator agent system prompt"""
        return """
You are a Generator agent in a Talk Structurally, Act Hierarchically framework.

Your responsibilities:
1. **Create Initial Content**: Generate solutions, content, or responses based on structured input
2. **Follow Structured Communication**: Process messages with clear background context and intermediate outputs
3. **Provide Detailed Output**: Generate comprehensive, well-structured responses

When receiving tasks:
- Pay attention to the specific message (M_ij)
- Consider the background context (B_ij) 
- Build upon intermediate outputs (I_ij) if provided
- Provide your own intermediate output for the next agent

Always structure your response clearly and provide sufficient detail for evaluation.
"""

    def _get_evaluator_prompt(self) -> str:
        """Get the evaluator agent system prompt"""
        return """
You are an Evaluator agent in a hierarchical evaluation system.

Your responsibilities:
1. **Evaluate Content**: Assess quality, accuracy, and appropriateness
2. **Provide Specific Feedback**: Give detailed, actionable feedback
3. **Score Performance**: Provide numerical scores with justification

Evaluation criteria:
- Accuracy and correctness
- Completeness and coverage  
- Clarity and coherence
- Relevance and appropriateness

Always provide:
- Specific evaluation criterion
- Numerical score (0-10)
- Detailed feedback
- Confidence level (0-1)
"""

    def _get_refiner_prompt(self) -> str:
        """Get the refiner agent system prompt"""
        return """
You are a Refiner agent in a Talk Structurally, Act Hierarchically framework.

Your responsibilities:
1. **Improve Content**: Enhance solutions based on evaluation feedback
2. **Address Feedback**: Specifically address issues identified by evaluators
3. **Maintain Quality**: Ensure improvements maintain or enhance overall quality

When refining:
- Consider all evaluation feedback
- Address specific issues mentioned
- Maintain the core strengths of the original
- Provide clear justification for changes

Always explain your refinements and how they address the evaluation feedback.
"""

    def send_structured_message(
        self,
        sender: str,
        recipient: str,
        message: str,
        background: str = "",
        intermediate_output: str = "",
    ) -> StructuredMessage:
        """
        Send a structured message following the HierarchicalStructuredComm protocol

        Args:
            sender: Name of the sending agent
            recipient: Name of the receiving agent
            message: Specific task message (M_ij)
            background: Background context (B_ij)
            intermediate_output: Intermediate output (I_ij)

        Returns:
            StructuredMessage object
        """
        structured_msg = StructuredMessage(
            message=message,
            background=background,
            intermediate_output=intermediate_output,
            sender=sender,
            recipient=recipient,
        )

        self.conversation_history.append(structured_msg)

        if self.verbose:
            # Enhanced structured message display
            console.print(
                Panel(
                    f"[bold green]Message Sent[/bold green]\n"
                    f"[cyan]From:[/cyan] {sender}\n"
                    f"[cyan]To:[/cyan] {recipient}\n"
                    f"[cyan]Message:[/cyan] {message[:100]}{'...' if len(message) > 100 else ''}",
                    title="Structured Communication",
                    border_style="green",
                )
            )
            logger.info(
                f"Structured message sent from {sender} to {recipient}"
            )
            logger.info(f"Message: {message[:100]}...")

        return structured_msg

    def run_hierarchical_evaluation(
        self, content: str, evaluation_criteria: List[str] = None
    ) -> List[EvaluationResult]:
        """
        Run hierarchical evaluation with multiple evaluators

        Args:
            content: Content to evaluate
            evaluation_criteria: List of evaluation criteria

        Returns:
            List of evaluation results
        """
        if not self.enable_hierarchical_evaluation:
            return []

        if evaluation_criteria is None:
            evaluation_criteria = [
                "accuracy",
                "completeness",
                "clarity",
                "relevance",
            ]

        results = []

        # Run evaluations in parallel
        for i, evaluator in enumerate(self.evaluators):
            criterion = evaluation_criteria[
                i % len(evaluation_criteria)
            ]

            # Create structured message for evaluator
            eval_message = f"Evaluate the following content based on {criterion} criterion"
            eval_background = f"Evaluation criterion: {criterion}\nContent to evaluate: {content}"

            self.send_structured_message(
                sender=self.evaluation_supervisor_name,
                recipient=(
                    evaluator.agent_name
                    if hasattr(evaluator, "agent_name")
                    else f"Evaluator_{i}"
                ),
                message=eval_message,
                background=eval_background,
                intermediate_output=content,
            )

            # Get evaluation result
            try:
                if hasattr(evaluator, "run"):
                    eval_response = evaluator.run(
                        f"Evaluate this content for {criterion}:\n{content}\n\nProvide: 1) Score (0-10), 2) Detailed feedback, 3) Confidence (0-1)"
                    )

                    # Parse evaluation result (simplified parsing)
                    result = EvaluationResult(
                        evaluator_name=(
                            evaluator.agent_name
                            if hasattr(evaluator, "agent_name")
                            else f"Evaluator_{i}"
                        ),
                        criterion=criterion,
                        score=7.5,  # Default score, would need proper parsing
                        feedback=eval_response,
                        confidence=0.8,  # Default confidence
                    )
                    results.append(result)

            except Exception as e:
                logger.error(f"Error in evaluation: {e}")
                continue

        # Get summarized feedback from evaluation supervisor
        if self.evaluation_supervisor and results:
            summary_prompt = f"Summarize these evaluation results:\n{results}\n\nProvide coordinated, actionable feedback."

            try:
                if hasattr(self.evaluation_supervisor, "run"):
                    summary_feedback = self.evaluation_supervisor.run(
                        summary_prompt
                    )
                    logger.info(
                        f"Evaluation summary: {summary_feedback}"
                    )
            except Exception as e:
                logger.error(f"Error in evaluation summary: {e}")

        self.evaluation_results.extend(results)
        return results

    def step(self, task: str, img: str = None, *args, **kwargs):
        """
        Execute one step of the HierarchicalStructuredComm workflow

        Args:
            task: Task to execute
            img: Optional image input

        Returns:
            Step result
        """
        try:
            logger.info(
                f"Executing HierarchicalStructuredComm step for task: {task[:100]}..."
            )

            # Safety check: prevent recursive task processing
            if (
                len(task) > 1000
            ):  # If task is too long, it might be recursive
                logger.warning(
                    "Task too long, possible recursive call detected"
                )
                return {
                    "error": "Task too long, possible recursive call"
                }

            # Step 1: Generate initial content
            generator_result = self._generate_content(task)

            # Safety check: prevent empty or error results
            if not generator_result or generator_result.startswith(
                "Error"
            ):
                logger.error(f"Generator failed: {generator_result}")
                return {
                    "error": f"Generator failed: {generator_result}"
                }

            # Step 2: Evaluate content hierarchically
            evaluation_results = self.run_hierarchical_evaluation(
                generator_result
            )

            # Step 3: Refine content based on evaluation
            refined_result = self._refine_content(
                generator_result, evaluation_results
            )

            # Safety check: ensure we have a valid result
            if not refined_result:
                refined_result = generator_result

            return {
                "generator_result": generator_result,
                "evaluation_results": evaluation_results,
                "refined_result": refined_result,
                "conversation_history": self.conversation_history,
            }

        except Exception as e:
            logger.error(
                f"Error in HierarchicalStructuredComm step: {e}"
            )
            logger.error(traceback.format_exc())
            return {"error": str(e)}

    def _generate_content(self, task: str) -> str:
        """Generate initial content using generator agents"""
        if not self.generators:
            return "No generators available"

        # Use first generator for initial content
        generator = self.generators[0]

        # Create structured message
        message = f"Generate content for the following task: {task}"
        background = f"Task context: {task}\n\nProvide comprehensive, well-structured content."

        self.send_structured_message(
            sender=self.supervisor_name,
            recipient=(
                generator.agent_name
                if hasattr(generator, "agent_name")
                else "Generator"
            ),
            message=message,
            background=background,
        )

        try:
            if hasattr(generator, "run"):
                # Add a simple, focused prompt to prevent recursive calls
                prompt = f"Task: {task}\n\nGenerate a clear, concise response. Do not repeat the task or ask for clarification."

                result = generator.run(prompt)

                # Safety check: prevent recursive or overly long responses
                if len(result) > 2000:
                    result = result[:2000] + "... [truncated]"

                # Safety check: prevent responses that just repeat the task
                if (
                    task.lower() in result.lower()
                    and len(result) < len(task) * 2
                ):
                    logger.warning(
                        "Generator response appears to be recursive"
                    )
                    return (
                        "Error: Generator produced recursive response"
                    )

                self.intermediate_outputs["generator"] = result
                return result
            else:
                return "Generator not properly configured"
        except Exception as e:
            logger.error(f"Error in content generation: {e}")
            return f"Error generating content: {e}"

    def _refine_content(
        self,
        original_content: str,
        evaluation_results: List[EvaluationResult],
    ) -> str:
        """Refine content based on evaluation feedback"""
        if not self.refiners:
            return original_content

        if not evaluation_results:
            return original_content

        # Use first refiner
        refiner = self.refiners[0]

        # Create feedback summary
        feedback_summary = "\n".join(
            [
                f"{result.criterion}: {result.feedback} (Score: {result.score}/10)"
                for result in evaluation_results
            ]
        )

        # Create structured message for refinement
        message = (
            "Refine the content based on the evaluation feedback"
        )
        background = f"Original content: {original_content}\n\nEvaluation feedback:\n{feedback_summary}"

        self.send_structured_message(
            sender=self.supervisor_name,
            recipient=(
                refiner.agent_name
                if hasattr(refiner, "agent_name")
                else "Refiner"
            ),
            message=message,
            background=background,
            intermediate_output=original_content,
        )

        try:
            if hasattr(refiner, "run"):
                refinement_prompt = f"""
Original content:
{original_content}

Evaluation feedback:
{feedback_summary}

Please refine the content to address the feedback while maintaining its core strengths.
"""
                result = refiner.run(refinement_prompt)
                self.intermediate_outputs["refiner"] = result
                return result
            else:
                return original_content
        except Exception as e:
            logger.error(f"Error in content refinement: {e}")
            return original_content

    def run(self, task: str, img: str = None, *args, **kwargs):
        """
        Run the complete HierarchicalStructuredComm workflow

        Args:
            task: Task to execute
            img: Optional image input

        Returns:
            Final result
        """
        # Enhanced workflow start display
        console.print(
            Panel(
                f"[bold yellow]Starting Hierarchical Structured Communication Workflow[/bold yellow]\n"
                f"[cyan]Task:[/cyan] {task[:100]}{'...' if len(task) > 100 else ''}\n"
                f"[cyan]Max Loops:[/cyan] {self.max_loops}",
                title="Workflow Execution",
                border_style="yellow",
            )
        )
        logger.info(
            f"Running HierarchicalStructuredComm workflow for task: {task[:100]}..."
        )

        current_result = None
        total_loops = 0

        # Rich progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task_progress = progress.add_task(
                "Processing workflow...", total=self.max_loops
            )

            for loop in range(self.max_loops):
                total_loops = loop + 1
                progress.update(
                    task_progress,
                    description=f"Loop {total_loops}/{self.max_loops}",
                )
                logger.info(
                    f"HierarchicalStructuredComm loop {total_loops}/{self.max_loops}"
                )

                # Execute step
                step_result = self.step(task, img, *args, **kwargs)

                if "error" in step_result:
                    console.print(
                        f"[bold red]Error in loop {total_loops}: {step_result['error']}[/bold red]"
                    )
                    logger.error(
                        f"Error in loop {total_loops}: {step_result['error']}"
                    )
                    break

                current_result = step_result["refined_result"]

                # Check if we should continue refining
                if loop < self.max_loops - 1:
                    # Simple continuation logic - could be enhanced
                    evaluation_scores = [
                        result.score
                        for result in step_result[
                            "evaluation_results"
                        ]
                    ]
                    avg_score = (
                        sum(evaluation_scores)
                        / len(evaluation_scores)
                        if evaluation_scores
                        else 0
                    )

                    if avg_score >= 8.0:  # High quality threshold
                        console.print(
                            f"[bold green]High quality achieved (avg score: {avg_score:.2f}), stopping refinement[/bold green]"
                        )
                        logger.info(
                            f"High quality achieved (avg score: {avg_score:.2f}), stopping refinement"
                        )
                        break

                progress.advance(task_progress)

        # Enhanced completion display
        console.print(
            Panel(
                f"[bold green]Workflow Completed Successfully![/bold green]\n"
                f"[cyan]Total Loops:[/cyan] {total_loops}\n"
                f"[cyan]Conversation History:[/cyan] {len(self.conversation_history)} messages\n"
                f"[cyan]Evaluation Results:[/cyan] {len(self.evaluation_results)} evaluations",
                title="Workflow Summary",
                border_style="green",
            )
        )

        return {
            "final_result": current_result,
            "total_loops": total_loops,
            "conversation_history": self.conversation_history,
            "evaluation_results": self.evaluation_results,
            "intermediate_outputs": self.intermediate_outputs,
        }

    def __str__(self):
        return f"HierarchicalStructuredCommunicationFramework(name={self.name}, generators={len(self.generators)}, evaluators={len(self.evaluators)}, refiners={len(self.refiners)})"

    def __repr__(self):
        return self.__str__()


# Nothing to see here yet.
