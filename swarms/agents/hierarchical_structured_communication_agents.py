"""
Specialized Agents for Hierarchical Structured Communication Framework Framework

This module provides specialized agent implementations for the HierarchicalStructuredCommunication framework,
including structured communication agents and hierarchical evaluation agents.
"""

from typing import Any, Dict, List, Optional, Union
from loguru import logger

from swarms.structs.agent import Agent
from swarms.schemas.talk_hierarchical_schemas import (
    StructuredMessageSchema,
    EvaluationResultSchema,
    GeneratorResponseSchema,
    EvaluatorResponseSchema,
    RefinerResponseSchema,
    CommunicationType,
    AgentRole
)


class HierarchicalStructuredCommunicationGenerator(Agent):
    """
    Generator agent for Hierarchical Structured Communication Framework framework
    
    This agent specializes in creating initial content following the structured
    communication protocol with Message (M_ij), Background (B_ij), and Intermediate Output (I_ij).
    """
    
    def __init__(
        self,
        agent_name: str = "TalkHierGenerator",
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        **kwargs
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
            **kwargs
        )
    
    def _get_default_generator_prompt(self) -> str:
        """Get the default system prompt for generator agents"""
        return """
You are a Generator agent in a Hierarchical Structured Communication Framework framework.

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
        **kwargs
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
            prompt = self._construct_structured_prompt(message, background, intermediate_output)
            
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
                confidence=0.0
            )
    
    def _construct_structured_prompt(
        self,
        message: str,
        background: str,
        intermediate_output: str
    ) -> str:
        """Construct a structured prompt for generation"""
        prompt_parts = []
        
        if message:
            prompt_parts.append(f"**Task Message (M_ij):** {message}")
        
        if background:
            prompt_parts.append(f"**Background Context (B_ij):** {background}")
        
        if intermediate_output:
            prompt_parts.append(f"**Intermediate Output (I_ij):** {intermediate_output}")
        
        prompt = "\n\n".join(prompt_parts)
        prompt += "\n\nPlease generate content following the structured response format."
        
        return prompt
    
    def _parse_generator_response(self, response: str) -> GeneratorResponseSchema:
        """Parse the generator response into structured format"""
        try:
            # Simple parsing - could be enhanced with more sophisticated parsing
            lines = response.split('\n')
            content = ""
            intermediate_output = ""
            reasoning = ""
            confidence = 0.8  # Default confidence
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.lower().startswith('content:'):
                    current_section = 'content'
                    content = line[8:].strip()
                elif line.lower().startswith('intermediate output:'):
                    current_section = 'intermediate'
                    intermediate_output = line[20:].strip()
                elif line.lower().startswith('reasoning:'):
                    current_section = 'reasoning'
                    reasoning = line[10:].strip()
                elif line.lower().startswith('confidence:'):
                    try:
                        confidence = float(line[11:].strip())
                    except ValueError:
                        confidence = 0.8
                elif current_section == 'content':
                    content += " " + line
                elif current_section == 'intermediate':
                    intermediate_output += " " + line
                elif current_section == 'reasoning':
                    reasoning += " " + line
            
            return GeneratorResponseSchema(
                content=content or response,
                intermediate_output=intermediate_output,
                reasoning=reasoning,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error parsing generator response: {e}")
            return GeneratorResponseSchema(
                content=response,
                intermediate_output="",
                reasoning="Error parsing response",
                confidence=0.5
            )


class HierarchicalStructuredCommunicationEvaluator(Agent):
    """
    Evaluator agent for Hierarchical Structured Communication Framework framework
    
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
        **kwargs
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
            system_prompt = self._get_default_evaluator_prompt(evaluation_criteria)
        
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            model_name=model_name,
            verbose=verbose,
            **kwargs
        )
        
        self.evaluation_criteria = evaluation_criteria or ["accuracy", "completeness", "clarity", "relevance"]
    
    def _get_default_evaluator_prompt(self, criteria: List[str] = None) -> str:
        """Get the default system prompt for evaluator agents"""
        if criteria is None:
            criteria = ["accuracy", "completeness", "clarity", "relevance"]
        
        criteria_text = "\n".join([f"- {criterion}" for criterion in criteria])
        
        return f"""
You are an Evaluator agent in a Hierarchical Structured Communication Framework framework.

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
        self,
        content: str,
        criterion: str,
        **kwargs
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
            prompt = self._construct_evaluation_prompt(content, criterion)
            
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
                suggestions=["Fix technical issues", "Retry evaluation"]
            )
    
    def _construct_evaluation_prompt(self, content: str, criterion: str) -> str:
        """Construct an evaluation prompt"""
        return f"""
**Content to Evaluate:**
{content}

**Evaluation Criterion:**
{criterion}

Please evaluate the content above based on the {criterion} criterion.

Provide your evaluation following the structured response format.
"""
    
    def _parse_evaluator_response(self, response: str, criterion: str) -> EvaluatorResponseSchema:
        """Parse the evaluator response into structured format"""
        try:
            lines = response.split('\n')
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
                
                if line.lower().startswith('score:'):
                    try:
                        score = float(line[6:].strip())
                    except ValueError:
                        score = 5.0
                elif line.lower().startswith('feedback:'):
                    current_section = 'feedback'
                    feedback = line[9:].strip()
                elif line.lower().startswith('confidence:'):
                    try:
                        confidence = float(line[11:].strip())
                    except ValueError:
                        confidence = 0.8
                elif line.lower().startswith('reasoning:'):
                    current_section = 'reasoning'
                    reasoning = line[10:].strip()
                elif line.lower().startswith('suggestions:'):
                    current_section = 'suggestions'
                elif current_section == 'feedback':
                    feedback += " " + line
                elif current_section == 'reasoning':
                    reasoning += " " + line
                elif current_section == 'suggestions':
                    if line.startswith('-') or line.startswith('•'):
                        suggestions.append(line[1:].strip())
                    else:
                        suggestions.append(line)
            
            return EvaluatorResponseSchema(
                criterion=criterion,
                score=score,
                feedback=feedback or "No feedback provided",
                confidence=confidence,
                reasoning=reasoning,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error parsing evaluator response: {e}")
            return EvaluatorResponseSchema(
                criterion=criterion,
                score=5.0,
                feedback="Error parsing evaluation response",
                confidence=0.0,
                reasoning="Error occurred during parsing",
                suggestions=["Fix parsing issues"]
            )


class HierarchicalStructuredCommunicationRefiner(Agent):
    """
    Refiner agent for Hierarchical Structured Communication Framework framework
    
    This agent specializes in improving content based on evaluation feedback
    and maintaining the structured communication protocol.
    """
    
    def __init__(
        self,
        agent_name: str = "TalkHierRefiner",
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        **kwargs
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
            **kwargs
        )
    
    def _get_default_refiner_prompt(self) -> str:
        """Get the default system prompt for refiner agents"""
        return """
You are a Refiner agent in a Hierarchical Structured Communication Framework framework.

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
        **kwargs
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
            prompt = self._construct_refinement_prompt(original_content, evaluation_results)
            
            # Get refinement response
            response = self.run(prompt, **kwargs)
            
            # Parse and structure response
            return self._parse_refiner_response(response, evaluation_results)
            
        except Exception as e:
            logger.error(f"Error in refinement: {e}")
            return RefinerResponseSchema(
                refined_content=original_content,
                changes_made=["Error occurred during refinement"],
                reasoning=f"Error during refinement: {e}",
                confidence=0.0,
                feedback_addressed=[]
            )
    
    def _construct_refinement_prompt(
        self,
        original_content: str,
        evaluation_results: List[EvaluationResultSchema]
    ) -> str:
        """Construct a refinement prompt"""
        feedback_summary = "\n\n".join([
            f"**{result.criterion} (Score: {result.score}/10):**\n{result.feedback}"
            for result in evaluation_results
        ])
        
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
        evaluation_results: List[EvaluationResultSchema]
    ) -> RefinerResponseSchema:
        """Parse the refiner response into structured format"""
        try:
            lines = response.split('\n')
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
                
                if line.lower().startswith('refined content:'):
                    current_section = 'content'
                    refined_content = line[16:].strip()
                elif line.lower().startswith('changes made:'):
                    current_section = 'changes'
                elif line.lower().startswith('reasoning:'):
                    current_section = 'reasoning'
                    reasoning = line[10:].strip()
                elif line.lower().startswith('confidence:'):
                    try:
                        confidence = float(line[11:].strip())
                    except ValueError:
                        confidence = 0.8
                elif line.lower().startswith('feedback addressed:'):
                    current_section = 'feedback'
                elif current_section == 'content':
                    refined_content += " " + line
                elif current_section == 'changes':
                    if line.startswith('-') or line.startswith('•'):
                        changes_made.append(line[1:].strip())
                    else:
                        changes_made.append(line)
                elif current_section == 'reasoning':
                    reasoning += " " + line
                elif current_section == 'feedback':
                    if line.startswith('-') or line.startswith('•'):
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
                feedback_addressed=feedback_addressed
            )
            
        except Exception as e:
            logger.error(f"Error parsing refiner response: {e}")
            return RefinerResponseSchema(
                refined_content=response,
                changes_made=["Error parsing response"],
                reasoning="Error occurred during parsing",
                confidence=0.0,
                feedback_addressed=[]
            )


class HierarchicalStructuredCommunicationSupervisor(Agent):
    """
    Supervisor agent for Hierarchical Structured Communication Framework framework
    
    This agent coordinates the overall workflow and manages structured communication
    between different agent types.
    """
    
    def __init__(
        self,
        agent_name: str = "TalkHierSupervisor",
        system_prompt: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        verbose: bool = False,
        **kwargs
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
            **kwargs
        )
    
    def _get_default_supervisor_prompt(self) -> str:
        """Get the default system prompt for supervisor agents"""
        return """
You are a Supervisor agent in a Hierarchical Structured Communication Framework framework.

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
        self,
        task: str,
        current_state: Dict[str, Any],
        **kwargs
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
            prompt = self._construct_coordination_prompt(task, current_state)
            
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
                "reasoning": "Error occurred during coordination"
            }
    
    def _construct_coordination_prompt(self, task: str, current_state: Dict[str, Any]) -> str:
        """Construct a coordination prompt"""
        state_summary = "\n".join([
            f"- {key}: {value}"
            for key, value in current_state.items()
        ])
        
        return f"""
**Current Task:**
{task}

**Current State:**
{state_summary}

Please coordinate the workflow and determine the next action.

Provide your coordination decision following the structured response format.
"""
    
    def _parse_coordination_response(self, response: str) -> Dict[str, Any]:
        """Parse the coordination response"""
        try:
            lines = response.split('\n')
            result = {
                "next_action": "continue",
                "target_agent": "generator",
                "structured_message": "",
                "background_context": "",
                "intermediate_output": "",
                "reasoning": ""
            }
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.lower().startswith('next action:'):
                    result["next_action"] = line[12:].strip()
                elif line.lower().startswith('target agent:'):
                    result["target_agent"] = line[13:].strip()
                elif line.lower().startswith('structured message:'):
                    current_section = 'message'
                    result["structured_message"] = line[19:].strip()
                elif line.lower().startswith('background context:'):
                    current_section = 'background'
                    result["background_context"] = line[19:].strip()
                elif line.lower().startswith('intermediate output:'):
                    current_section = 'output'
                    result["intermediate_output"] = line[20:].strip()
                elif line.lower().startswith('reasoning:'):
                    current_section = 'reasoning'
                    result["reasoning"] = line[10:].strip()
                elif current_section == 'message':
                    result["structured_message"] += " " + line
                elif current_section == 'background':
                    result["background_context"] += " " + line
                elif current_section == 'output':
                    result["intermediate_output"] += " " + line
                elif current_section == 'reasoning':
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
                "reasoning": "Error occurred during parsing"
            } 
