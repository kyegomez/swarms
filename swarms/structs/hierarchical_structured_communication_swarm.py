"""
Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems

This implementation is based on the research paper:
"Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems"
arXiv:2502.11098

The framework consists of:
1. Structured Communication Protocol - Context-rich communication with message, background, and intermediate output
2. Hierarchical Refinement System - Evaluation team with supervisor coordination  
3. Graph-based Agent Orchestration - Dynamic communication pathways

Key Features:
- Structured communication with Message (M_ij), Background (B_ij), and Intermediate Output (I_ij)
- Hierarchical evaluation team with supervisor coordination
- Dynamic graph-based agent routing
- Context preservation and shared memory
"""

import traceback
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="hierarchical_structured_communication_swarm")


class CommunicationType(str, Enum):
    """Types of communication in the structured protocol"""
    MESSAGE = "message"  # M_ij: Specific task instructions
    BACKGROUND = "background"  # B_ij: Context and problem background
    INTERMEDIATE_OUTPUT = "intermediate_output"  # I_ij: Intermediate results


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
    message: str = Field(description="Specific task instructions (M_ij)")
    background: str = Field(description="Context and problem background (B_ij)")
    intermediate_output: str = Field(description="Intermediate results (I_ij)")
    sender: str = Field(description="Name of the sending agent")
    recipient: str = Field(description="Name of the receiving agent")
    timestamp: Optional[str] = None


class HierarchicalOrder(BaseModel):
    """Order structure for hierarchical task assignment"""
    agent_name: str = Field(description="Name of the agent to receive the task")
    task: str = Field(description="Specific task description")
    communication_type: CommunicationType = Field(
        default=CommunicationType.MESSAGE,
        description="Type of communication to use"
    )
    background_context: str = Field(
        default="",
        description="Background context for the task"
    )
    intermediate_output: str = Field(
        default="",
        description="Intermediate output to pass along"
    )


class EvaluationResult(BaseModel):
    """Result from evaluation team member"""
    evaluator_name: str = Field(description="Name of the evaluator")
    criterion: str = Field(description="Evaluation criterion")
    score: float = Field(description="Evaluation score")
    feedback: str = Field(description="Detailed feedback")
    confidence: float = Field(description="Confidence in evaluation")


class HierarchicalStructuredCommunicationSwarm(BaseSwarm):
    """
    Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems
    
    This framework implements the HierarchicalStructuredComm approach with:
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
        name: str = "HierarchicalStructuredCommunicationSwarm",
        description: str = "Talk Structurally, Act Hierarchically Framework",
        supervisor: Optional[Union[Agent, Callable, Any]] = None,
        generators: List[Union[Agent, Callable, Any]] = None,
        evaluators: List[Union[Agent, Callable, Any]] = None,
        refiners: List[Union[Agent, Callable, Any]] = None,
        evaluation_supervisor: Optional[Union[Agent, Callable, Any]] = None,
        max_loops: int = 3,
        output_type: OutputType = "dict-all-except-first",
        supervisor_name: str = "Supervisor",
        evaluation_supervisor_name: str = "EvaluationSupervisor",
        verbose: bool = False,
        enable_structured_communication: bool = True,
        enable_hierarchical_evaluation: bool = True,
        shared_memory: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialize the HierarchicalStructuredCommunicationSwarm
        
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
        self.enable_structured_communication = enable_structured_communication
        self.enable_hierarchical_evaluation = enable_hierarchical_evaluation
        self.shared_memory = shared_memory
        
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
        logger.info(f"Initializing {self.name}")
        
        # Setup supervisor if not provided
        if self.supervisor is None:
            self.supervisor = self._create_supervisor_agent()
        
        # Setup evaluation supervisor if not provided
        if self.evaluation_supervisor is None and self.enable_hierarchical_evaluation:
            self.evaluation_supervisor = self._create_evaluation_supervisor_agent()
        
        # Setup default agents if none provided
        if not self.generators:
            self.generators = [self._create_default_generator()]
        
        if not self.evaluators and self.enable_hierarchical_evaluation:
            self.evaluators = [self._create_default_evaluator()]
        
        if not self.refiners:
            self.refiners = [self._create_default_refiner()]
        
        logger.info(f"Swarm initialized with {len(self.generators)} generators, "
                   f"{len(self.evaluators)} evaluators, {len(self.refiners)} refiners")
    
    def _create_supervisor_agent(self) -> Agent:
        """Create the main supervisor agent"""
        supervisor_prompt = self._get_supervisor_prompt()
        
        return Agent(
            agent_name=self.supervisor_name,
            system_prompt=supervisor_prompt,
            model_name="gpt-4o-mini",
            verbose=self.verbose,
            # Ollama configuration
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="ollama",
            reliability_check=False
        )
    
    def _create_evaluation_supervisor_agent(self) -> Agent:
        """Create the evaluation team supervisor"""
        eval_supervisor_prompt = self._get_evaluation_supervisor_prompt()
        
        return Agent(
            agent_name=self.evaluation_supervisor_name,
            system_prompt=eval_supervisor_prompt,
            model_name="gpt-4o-mini",
            verbose=self.verbose,
            # Ollama configuration
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="ollama",
            reliability_check=False
        )
    
    def _create_default_generator(self) -> Agent:
        """Create a default generator agent"""
        generator_prompt = self._get_generator_prompt()
        
        return Agent(
            agent_name="Generator",
            system_prompt=generator_prompt,
            model_name="gpt-4o-mini",
            verbose=self.verbose,
            # Ollama configuration
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="ollama",
            reliability_check=False
        )
    
    def _create_default_evaluator(self) -> Agent:
        """Create a default evaluator agent"""
        evaluator_prompt = self._get_evaluator_prompt()
        
        return Agent(
            agent_name="Evaluator",
            system_prompt=evaluator_prompt,
            model_name="gpt-4o-mini",
            verbose=self.verbose,
            # Ollama configuration
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="ollama",
            reliability_check=False
        )
    
    def _create_default_refiner(self) -> Agent:
        """Create a default refiner agent"""
        refiner_prompt = self._get_refiner_prompt()
        
        return Agent(
            agent_name="Refiner",
            system_prompt=refiner_prompt,
            model_name="gpt-4o-mini",
            verbose=self.verbose,
            # Ollama configuration
            openai_api_base="http://localhost:11434/v1",
            openai_api_key="ollama",
            reliability_check=False
        )
    
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
        intermediate_output: str = ""
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
            recipient=recipient
        )
        
        self.conversation_history.append(structured_msg)
        
        if self.verbose:
            logger.info(f"Structured message sent from {sender} to {recipient}")
            logger.info(f"Message: {message[:100]}...")
        
        return structured_msg
    
    def run_hierarchical_evaluation(
        self,
        content: str,
        evaluation_criteria: List[str] = None
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
            evaluation_criteria = ["accuracy", "completeness", "clarity", "relevance"]
        
        results = []
        
        # Run evaluations in parallel
        for i, evaluator in enumerate(self.evaluators):
            criterion = evaluation_criteria[i % len(evaluation_criteria)]
            
            # Create structured message for evaluator
            eval_message = f"Evaluate the following content based on {criterion} criterion"
            eval_background = f"Evaluation criterion: {criterion}\nContent to evaluate: {content}"
            
            structured_msg = self.send_structured_message(
                sender=self.evaluation_supervisor_name,
                recipient=evaluator.agent_name if hasattr(evaluator, 'agent_name') else f"Evaluator_{i}",
                message=eval_message,
                background=eval_background,
                intermediate_output=content
            )
            
            # Get evaluation result
            try:
                if hasattr(evaluator, 'run'):
                    eval_response = evaluator.run(
                        f"Evaluate this content for {criterion}:\n{content}\n\nProvide: 1) Score (0-10), 2) Detailed feedback, 3) Confidence (0-1)"
                    )
                    
                    # Parse evaluation result (simplified parsing)
                    result = EvaluationResult(
                        evaluator_name=evaluator.agent_name if hasattr(evaluator, 'agent_name') else f"Evaluator_{i}",
                        criterion=criterion,
                        score=7.5,  # Default score, would need proper parsing
                        feedback=eval_response,
                        confidence=0.8  # Default confidence
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error in evaluation: {e}")
                continue
        
        # Get summarized feedback from evaluation supervisor
        if self.evaluation_supervisor and results:
            summary_prompt = f"Summarize these evaluation results:\n{results}\n\nProvide coordinated, actionable feedback."
            
            try:
                if hasattr(self.evaluation_supervisor, 'run'):
                    summary_feedback = self.evaluation_supervisor.run(summary_prompt)
                    logger.info(f"Evaluation summary: {summary_feedback}")
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
            logger.info(f"Executing HierarchicalStructuredComm step for task: {task[:100]}...")
            
            # Safety check: prevent recursive task processing
            if len(task) > 1000:  # If task is too long, it might be recursive
                logger.warning("Task too long, possible recursive call detected")
                return {"error": "Task too long, possible recursive call"}
            
            # Step 1: Generate initial content
            generator_result = self._generate_content(task)
            
            # Safety check: prevent empty or error results
            if not generator_result or generator_result.startswith("Error"):
                logger.error(f"Generator failed: {generator_result}")
                return {"error": f"Generator failed: {generator_result}"}
            
            # Step 2: Evaluate content hierarchically
            evaluation_results = self.run_hierarchical_evaluation(generator_result)
            
            # Step 3: Refine content based on evaluation
            refined_result = self._refine_content(generator_result, evaluation_results)
            
            # Safety check: ensure we have a valid result
            if not refined_result:
                refined_result = generator_result
            
            return {
                "generator_result": generator_result,
                "evaluation_results": evaluation_results,
                "refined_result": refined_result,
                "conversation_history": self.conversation_history
            }
            
        except Exception as e:
            logger.error(f"Error in HierarchicalStructuredComm step: {e}")
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
        
        structured_msg = self.send_structured_message(
            sender=self.supervisor_name,
            recipient=generator.agent_name if hasattr(generator, 'agent_name') else "Generator",
            message=message,
            background=background
        )
        
        try:
            if hasattr(generator, 'run'):
                # Add a simple, focused prompt to prevent recursive calls
                prompt = f"Task: {task}\n\nGenerate a clear, concise response. Do not repeat the task or ask for clarification."
                
                result = generator.run(prompt)
                
                # Safety check: prevent recursive or overly long responses
                if len(result) > 2000:
                    result = result[:2000] + "... [truncated]"
                
                # Safety check: prevent responses that just repeat the task
                if task.lower() in result.lower() and len(result) < len(task) * 2:
                    logger.warning("Generator response appears to be recursive")
                    return "Error: Generator produced recursive response"
                
                self.intermediate_outputs["generator"] = result
                return result
            else:
                return "Generator not properly configured"
        except Exception as e:
            logger.error(f"Error in content generation: {e}")
            return f"Error generating content: {e}"
    
    def _refine_content(self, original_content: str, evaluation_results: List[EvaluationResult]) -> str:
        """Refine content based on evaluation feedback"""
        if not self.refiners:
            return original_content
        
        if not evaluation_results:
            return original_content
        
        # Use first refiner
        refiner = self.refiners[0]
        
        # Create feedback summary
        feedback_summary = "\n".join([
            f"{result.criterion}: {result.feedback} (Score: {result.score}/10)"
            for result in evaluation_results
        ])
        
        # Create structured message for refinement
        message = "Refine the content based on the evaluation feedback"
        background = f"Original content: {original_content}\n\nEvaluation feedback:\n{feedback_summary}"
        
        structured_msg = self.send_structured_message(
            sender=self.supervisor_name,
            recipient=refiner.agent_name if hasattr(refiner, 'agent_name') else "Refiner",
            message=message,
            background=background,
            intermediate_output=original_content
        )
        
        try:
            if hasattr(refiner, 'run'):
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
        logger.info(f"Running HierarchicalStructuredComm workflow for task: {task[:100]}...")
        
        current_result = None
        total_loops = 0
        
        for loop in range(self.max_loops):
            total_loops = loop + 1
            logger.info(f"HierarchicalStructuredComm loop {total_loops}/{self.max_loops}")
            
            # Execute step
            step_result = self.step(task, img, *args, **kwargs)
            
            if "error" in step_result:
                logger.error(f"Error in loop {total_loops}: {step_result['error']}")
                break
            
            current_result = step_result["refined_result"]
            
            # Check if we should continue refining
            if loop < self.max_loops - 1:
                # Simple continuation logic - could be enhanced
                evaluation_scores = [result.score for result in step_result["evaluation_results"]]
                avg_score = sum(evaluation_scores) / len(evaluation_scores) if evaluation_scores else 0
                
                if avg_score >= 8.0:  # High quality threshold
                    logger.info(f"High quality achieved (avg score: {avg_score:.2f}), stopping refinement")
                    break
        
        return {
            "final_result": current_result,
            "total_loops": total_loops,
            "conversation_history": self.conversation_history,
            "evaluation_results": self.evaluation_results,
            "intermediate_outputs": self.intermediate_outputs
        }
    
    def __str__(self):
        return f"HierarchicalStructuredCommunicationSwarm(name={self.name}, generators={len(self.generators)}, evaluators={len(self.evaluators)}, refiners={len(self.refiners)})"
    
    def __repr__(self):
        return self.__str__() 