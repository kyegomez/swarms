import os
from typing import List, Dict, Any, Optional
from loguru import logger
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.utils.function_caller_model import OpenAIFunctionCaller
from swarms.structs.ma_utils import set_random_models_for_agents
from swarms.structs.swarm_router import SwarmRouter, SwarmRouterConfig
from swarms.structs.council_judge import CouncilAsAJudge
from dotenv import load_dotenv


load_dotenv()

BOSS_SYSTEM_PROMPT = """
You are an expert swarm manager and agent architect. Your role is to create and coordinate a team of specialized AI agents, each with distinct personalities, roles, and capabilities. Your primary goal is to ensure the swarm operates efficiently while maintaining clear communication and well-defined responsibilities.

### Core Principles:

1. **Deep Task Understanding**:
   - First, thoroughly analyze the task requirements, breaking them down into core components and sub-tasks
   - Identify the necessary skills, knowledge domains, and personality traits needed for each component
   - Consider potential challenges, dependencies, and required coordination between agents
   - Map out the ideal workflow and information flow between agents

2. **Agent Design Philosophy**:
   - Each agent must have a clear, specific purpose and domain of expertise
   - Agents should have distinct personalities that complement their roles
   - Design agents to be self-aware of their limitations and when to seek help
   - Ensure agents can effectively communicate their progress and challenges

3. **Agent Creation Framework**:
   For each new agent, define:
   - **Role & Purpose**: Clear, specific description of what the agent does and why
   - **Personality Traits**: Distinct characteristics that influence how the agent thinks and communicates
   - **Expertise Level**: Specific knowledge domains and skill sets
   - **Communication Style**: How the agent presents information and interacts
   - **Decision-Making Process**: How the agent approaches problems and makes choices
   - **Limitations & Boundaries**: What the agent cannot or should not do
   - **Collaboration Protocol**: How the agent works with others

4. **System Prompt Design**:
   Create detailed system prompts that include:
   - Role and purpose explanation
   - Personality description and behavioral guidelines
   - Specific capabilities and tools available
   - Communication protocols and reporting requirements
   - Problem-solving approach and decision-making framework
   - Collaboration guidelines and team interaction rules
   - Quality standards and success criteria

5. **Swarm Coordination**:
   - Design clear communication channels between agents
   - Establish protocols for task handoffs and information sharing
   - Create feedback loops for continuous improvement
   - Implement error handling and recovery procedures
   - Define escalation paths for complex issues

6. **Quality Assurance**:
   - Set clear success criteria for each agent and the overall swarm
   - Implement verification steps for task completion
   - Create mechanisms for self-assessment and improvement
   - Establish protocols for handling edge cases and unexpected situations

### Output Format:

When creating a new agent or swarm, provide:

1. **Agent Design**:
   - Role and purpose statement
   - Personality profile
   - Capabilities and limitations
   - Communication style
   - Collaboration protocols

2. **System Prompt**:
   - Complete, detailed prompt that embodies the agent's identity
   - Clear instructions for behavior and decision-making
   - Specific guidelines for interaction and reporting

3. **Swarm Architecture**:
   - Team structure and hierarchy
   - Communication flow
   - Task distribution plan
   - Quality control measures

### Notes:

- Always prioritize clarity and specificity in agent design
- Ensure each agent has a unique, well-defined role
- Create detailed, comprehensive system prompts
- Maintain clear documentation of agent capabilities and limitations
- Design for scalability and adaptability
- Focus on creating agents that can work together effectively
- Consider edge cases and potential failure modes
- Implement robust error handling and recovery procedures
"""


class AgentConfig(BaseModel):
    """Configuration for an individual agent in a swarm"""

    name: str = Field(
        description="The name of the agent",
    )
    description: str = Field(
        description="A description of the agent's purpose and capabilities",
    )
    system_prompt: str = Field(
        description="The system prompt that defines the agent's behavior",
    )

    # max_loops: int = Field(
    #     description="The maximum number of loops for the agent to run",
    # )

    class Config:
        arbitrary_types_allowed = True


class AgentsConfig(BaseModel):
    """Configuration for a list of agents in a swarm"""

    agents: List[AgentConfig] = Field(
        description="A list of agent configurations",
    )


class EvaluationResult(BaseModel):
    """Results from evaluating a swarm iteration"""
    
    iteration: int = Field(description="The iteration number")
    task: str = Field(description="The original task")
    output: str = Field(description="The swarm's output")
    evaluation_scores: Dict[str, float] = Field(description="Scores for different evaluation dimensions")
    feedback: str = Field(description="Detailed feedback from evaluation")
    strengths: List[str] = Field(description="Identified strengths")
    weaknesses: List[str] = Field(description="Identified weaknesses")
    suggestions: List[str] = Field(description="Suggestions for improvement")


class IterativeImprovementConfig(BaseModel):
    """Configuration for iterative improvement process"""
    
    max_iterations: int = Field(default=3, description="Maximum number of improvement iterations")
    improvement_threshold: float = Field(default=0.1, description="Minimum improvement required to continue")
    evaluation_dimensions: List[str] = Field(
        default=["accuracy", "helpfulness", "coherence", "instruction_adherence"],
        description="Dimensions to evaluate"
    )
    use_judge_agent: bool = Field(default=True, description="Whether to use CouncilAsAJudge for evaluation")
    store_all_iterations: bool = Field(default=True, description="Whether to store results from all iterations")


class AutoSwarmBuilder:
    """A class that automatically builds and manages swarms of AI agents with autonomous evaluation.

    This class handles the creation, coordination and execution of multiple AI agents working
    together as a swarm to accomplish complex tasks. It uses a boss agent to delegate work
    and create new specialized agents as needed. The autonomous evaluation feature allows
    for iterative improvement of agent performance through feedback loops.

    Args:
        name (str): The name of the swarm
        description (str): A description of the swarm's purpose
        verbose (bool, optional): Whether to output detailed logs. Defaults to True.
        max_loops (int, optional): Maximum number of execution loops. Defaults to 1.
        random_models (bool, optional): Whether to use random models for agents. Defaults to True.
        enable_evaluation (bool, optional): Whether to enable autonomous evaluation. Defaults to False.
        evaluation_config (IterativeImprovementConfig, optional): Configuration for evaluation process.
    """

    def __init__(
        self,
        name: str = None,
        description: str = None,
        verbose: bool = True,
        max_loops: int = 1,
        random_models: bool = True,
        enable_evaluation: bool = False,
        evaluation_config: Optional[IterativeImprovementConfig] = None,
    ):
        """Initialize the AutoSwarmBuilder.

        Args:
            name (str): The name of the swarm
            description (str): A description of the swarm's purpose
            verbose (bool): Whether to output detailed logs
            max_loops (int): Maximum number of execution loops
            random_models (bool): Whether to use random models for agents
            enable_evaluation (bool): Whether to enable autonomous evaluation
            evaluation_config (IterativeImprovementConfig): Configuration for evaluation process
        """
        self.name = name
        self.description = description
        self.verbose = verbose
        self.max_loops = max_loops
        self.random_models = random_models
        self.enable_evaluation = enable_evaluation
        self.evaluation_config = evaluation_config or IterativeImprovementConfig()
        
        # Store evaluation history
        self.evaluation_history: List[EvaluationResult] = []
        self.current_iteration = 0
        
        # Initialize evaluation agents
        if self.enable_evaluation:
            self._initialize_evaluation_system()

        logger.info(
            f"Initializing AutoSwarmBuilder with name: {name}, description: {description}, "
            f"evaluation enabled: {enable_evaluation}"
        )

    def _initialize_evaluation_system(self):
        """Initialize the evaluation system with judge agents and evaluators"""
        try:
            # Initialize the council of judges for comprehensive evaluation
            if self.evaluation_config.use_judge_agent:
                self.council_judge = CouncilAsAJudge(
                    name="SwarmEvaluationCouncil",
                    description="Evaluates swarm performance across multiple dimensions",
                    model_name="gpt-4o-mini",
                    aggregation_model_name="gpt-4o-mini",
                )
            
            # Initialize improvement strategist agent
            self.improvement_agent = Agent(
                agent_name="ImprovementStrategist",
                description="Analyzes evaluation feedback and suggests agent improvements",
                system_prompt=self._get_improvement_agent_prompt(),
                model_name="gpt-4o-mini",
                max_loops=1,
                dynamic_temperature_enabled=True,
            )
            
            logger.info("Evaluation system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize evaluation system: {str(e)}")
            self.enable_evaluation = False
            raise

    def _get_improvement_agent_prompt(self) -> str:
        """Get the system prompt for the improvement strategist agent"""
        return """You are an expert AI swarm improvement strategist. Your role is to analyze evaluation feedback 
        and suggest specific improvements for agent configurations, roles, and coordination.

        Your responsibilities:
        1. Analyze evaluation feedback across multiple dimensions (accuracy, helpfulness, coherence, etc.)
        2. Identify patterns in agent performance and coordination issues
        3. Suggest specific improvements to agent roles, system prompts, and swarm architecture
        4. Recommend changes to agent collaboration protocols and task distribution
        5. Provide actionable recommendations for the next iteration

        When analyzing feedback, focus on:
        - Role clarity and specialization
        - Communication and coordination patterns
        - Task distribution effectiveness
        - Knowledge gaps or redundancies
        - Workflow optimization opportunities

        Provide your recommendations in a structured format:
        1. Key Issues Identified
        2. Specific Agent Improvements (per agent)
        3. Swarm Architecture Changes
        4. Coordination Protocol Updates
        5. Priority Ranking of Changes

        Be specific and actionable in your recommendations."""

    def run(self, task: str, *args, **kwargs):
        """Run the swarm on a given task with optional autonomous evaluation.

        Args:
            task (str): The task to execute
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Any: The result of the swarm execution (final iteration if evaluation enabled)

        Raises:
            Exception: If there's an error during execution
        """
        try:
            logger.info(f"Starting swarm execution for task: {task}")
            
            if not self.enable_evaluation:
                # Standard execution without evaluation
                return self._run_single_iteration(task, *args, **kwargs)
            else:
                # Autonomous evaluation enabled - run iterative improvement
                return self._run_with_autonomous_evaluation(task, *args, **kwargs)
                
        except Exception as e:
            logger.error(f"Error in swarm execution: {str(e)}", exc_info=True)
            raise

    def _run_single_iteration(self, task: str, *args, **kwargs):
        """Run a single iteration without evaluation"""
        agents = self.create_agents(task)
        logger.info(f"Created {len(agents)} agents")

        if self.random_models:
            logger.info("Setting random models for agents")
            agents = set_random_models_for_agents(agents=agents)

        return self.initialize_swarm_router(agents=agents, task=task)

    def _run_with_autonomous_evaluation(self, task: str, *args, **kwargs):
        """Run with autonomous evaluation and iterative improvement"""
        logger.info(f"Starting autonomous evaluation process for task: {task}")
        
        best_result = None
        best_score = 0.0
        
        for iteration in range(self.evaluation_config.max_iterations):
                         self.current_iteration = iteration + 1
             logger.info(f"Starting iteration {self.current_iteration}/{self.evaluation_config.max_iterations}")
             
             # Create agents (using feedback from previous iterations if available)
             agents = self.create_agents_with_feedback(task)
             
             if self.random_models and agents:
                 agents = set_random_models_for_agents(agents=agents)
             
             # Execute the swarm
             result = self.initialize_swarm_router(agents=agents, task=task)
             
             # Evaluate the result
             evaluation = self._evaluate_swarm_output(task, result, agents)
            
            # Store evaluation results
            self.evaluation_history.append(evaluation)
            
            # Calculate overall score
            overall_score = sum(evaluation.evaluation_scores.values()) / len(evaluation.evaluation_scores)
            
            logger.info(f"Iteration {self.current_iteration} overall score: {overall_score:.3f}")
            
            # Update best result if this iteration is better
            if overall_score > best_score:
                best_score = overall_score
                best_result = result
                logger.info(f"New best result found in iteration {self.current_iteration}")
            
            # Check if we should continue iterating
            if iteration > 0:
                improvement = overall_score - sum(self.evaluation_history[-2].evaluation_scores.values()) / len(self.evaluation_history[-2].evaluation_scores)
                if improvement < self.evaluation_config.improvement_threshold:
                    logger.info(f"Improvement threshold not met ({improvement:.3f} < {self.evaluation_config.improvement_threshold}). Stopping iterations.")
                    break
        
        # Log final results
        self._log_evaluation_summary()
        
        return best_result

    def _evaluate_swarm_output(self, task: str, output: str, agents: List[Agent]) -> EvaluationResult:
        """Evaluate the output of a swarm iteration"""
        try:
            logger.info(f"Evaluating swarm output for iteration {self.current_iteration}")
            
            # Use CouncilAsAJudge for comprehensive evaluation
            evaluation_scores = {}
            detailed_feedback = ""
            
            if self.evaluation_config.use_judge_agent:
                # Set up a base agent for the council to evaluate
                base_agent = Agent(
                    agent_name="SwarmOutput",
                    description="Combined output from the swarm",
                    system_prompt="You are representing the collective output of a swarm",
                    model_name="gpt-4o-mini",
                    max_loops=1,
                )
                
                # Configure the council judge with our base agent
                self.council_judge.base_agent = base_agent
                
                # Run evaluation
                evaluation_result = self.council_judge.run(task, output)
                detailed_feedback = str(evaluation_result)
                
                # Extract scores from the evaluation (simplified scoring based on feedback)
                for dimension in self.evaluation_config.evaluation_dimensions:
                    # Simple scoring based on presence of positive indicators in feedback
                    score = self._extract_dimension_score(detailed_feedback, dimension)
                    evaluation_scores[dimension] = score
            
            # Generate improvement suggestions
            improvement_feedback = self._generate_improvement_suggestions(
                task, output, detailed_feedback, agents
            )
            
            # Parse strengths, weaknesses, and suggestions from feedback
            strengths, weaknesses, suggestions = self._parse_feedback(improvement_feedback)
            
            return EvaluationResult(
                iteration=self.current_iteration,
                task=task,
                output=output,
                evaluation_scores=evaluation_scores,
                feedback=detailed_feedback,
                strengths=strengths,
                weaknesses=weaknesses,
                suggestions=suggestions,
            )
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            # Return a basic evaluation result in case of error
            return EvaluationResult(
                iteration=self.current_iteration,
                task=task,
                output=output,
                evaluation_scores={dim: 0.5 for dim in self.evaluation_config.evaluation_dimensions},
                feedback=f"Evaluation error: {str(e)}",
                strengths=[],
                weaknesses=["Evaluation system error"],
                suggestions=["Review evaluation system configuration"],
            )

    def _extract_dimension_score(self, feedback: str, dimension: str) -> float:
        """Extract a numerical score for a dimension from textual feedback"""
        # Simple heuristic scoring based on keyword presence
        positive_keywords = {
            "accuracy": ["accurate", "correct", "factual", "precise", "reliable"],
            "helpfulness": ["helpful", "useful", "practical", "actionable", "valuable"],
            "coherence": ["coherent", "logical", "structured", "organized", "clear"],
            "instruction_adherence": ["follows", "adheres", "complies", "meets requirements", "addresses"],
        }
        
        negative_keywords = {
            "accuracy": ["inaccurate", "incorrect", "wrong", "false", "misleading"],
            "helpfulness": ["unhelpful", "useless", "impractical", "vague", "unclear"],
            "coherence": ["incoherent", "confusing", "disorganized", "unclear", "jumbled"],
            "instruction_adherence": ["ignores", "fails to", "misses", "incomplete", "off-topic"],
        }
        
        feedback_lower = feedback.lower()
        
        positive_count = sum(1 for keyword in positive_keywords.get(dimension, []) if keyword in feedback_lower)
        negative_count = sum(1 for keyword in negative_keywords.get(dimension, []) if keyword in feedback_lower)
        
        # Calculate score (0.0 to 1.0)
        if positive_count + negative_count == 0:
            return 0.5  # Neutral if no keywords found
        
        score = positive_count / (positive_count + negative_count)
        return max(0.0, min(1.0, score))

    def _generate_improvement_suggestions(
        self, task: str, output: str, evaluation_feedback: str, agents: List[Agent]
    ) -> str:
        """Generate specific improvement suggestions based on evaluation"""
        try:
            agent_info = "\n".join([
                f"Agent: {agent.agent_name} - {agent.description}" 
                for agent in agents
            ])
            
            improvement_prompt = f"""
            Analyze the following swarm execution and provide specific improvement recommendations:
            
            Task: {task}
            
            Current Agents:
            {agent_info}
            
            Swarm Output: {output}
            
            Evaluation Feedback: {evaluation_feedback}
            
            Previous Iterations: {len(self.evaluation_history)} completed
            
            Provide specific, actionable recommendations for improving the swarm in the next iteration.
            """
            
            return self.improvement_agent.run(improvement_prompt)
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {str(e)}")
            return "Unable to generate improvement suggestions due to error."

    def _parse_feedback(self, feedback: str) -> tuple[List[str], List[str], List[str]]:
        """Parse feedback into strengths, weaknesses, and suggestions"""
        # Simple parsing logic - in practice, could be more sophisticated
        strengths = []
        weaknesses = []
        suggestions = []
        
        lines = feedback.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(keyword in line.lower() for keyword in ['strength', 'positive', 'good', 'well']):
                current_section = 'strengths'
                strengths.append(line)
            elif any(keyword in line.lower() for keyword in ['weakness', 'issue', 'problem', 'poor']):
                current_section = 'weaknesses'
                weaknesses.append(line)
            elif any(keyword in line.lower() for keyword in ['suggest', 'recommend', 'improve', 'should']):
                current_section = 'suggestions'
                suggestions.append(line)
            elif current_section == 'strengths' and line.startswith(('-', '•', '*')):
                strengths.append(line)
            elif current_section == 'weaknesses' and line.startswith(('-', '•', '*')):
                weaknesses.append(line)
            elif current_section == 'suggestions' and line.startswith(('-', '•', '*')):
                suggestions.append(line)
        
        return strengths[:5], weaknesses[:5], suggestions[:5]  # Limit to top 5 each

    def create_agents_with_feedback(self, task: str) -> List[Agent]:
        """Create agents incorporating feedback from previous iterations"""
        if not self.evaluation_history:
            # First iteration - use standard agent creation
            return self.create_agents(task)
        
        try:
            logger.info("Creating agents with feedback from previous iterations")
            
            # Get the latest evaluation feedback
            latest_evaluation = self.evaluation_history[-1]
            
            # Create enhanced prompt that includes improvement suggestions
            enhanced_task_prompt = f"""
            Original Task: {task}
            
            Previous Iteration Feedback:
            Strengths: {'; '.join(latest_evaluation.strengths)}
            Weaknesses: {'; '.join(latest_evaluation.weaknesses)}
            Suggestions: {'; '.join(latest_evaluation.suggestions)}
            
            Based on this feedback, create an improved set of agents that addresses the identified weaknesses
            and builds upon the strengths. Focus on the specific suggestions provided.
            
            Create agents for: {task}
            """
            
            model = OpenAIFunctionCaller(
                system_prompt=BOSS_SYSTEM_PROMPT,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.5,
                base_model=AgentsConfig,
            )

            logger.info("Getting improved agent configurations from boss agent")
            output = model.run(enhanced_task_prompt)
            logger.debug(f"Received improved agent configurations: {output.model_dump()}")
            output = output.model_dump()

            agents = []
            if isinstance(output, dict):
                for agent_config in output["agents"]:
                    logger.info(f"Building improved agent: {agent_config['name']}")
                    agent = self.build_agent(
                        agent_name=agent_config["name"],
                        agent_description=agent_config["description"],
                        agent_system_prompt=agent_config["system_prompt"],
                    )
                    agents.append(agent)
                    logger.info(f"Successfully built improved agent: {agent_config['name']}")

            return agents
            
        except Exception as e:
            logger.error(f"Error creating agents with feedback: {str(e)}")
            # Fallback to standard agent creation
            return self.create_agents(task)

    def _log_evaluation_summary(self):
        """Log a summary of all evaluation iterations"""
        if not self.evaluation_history:
            return
            
        logger.info("=== EVALUATION SUMMARY ===")
        logger.info(f"Total iterations: {len(self.evaluation_history)}")
        
        for i, evaluation in enumerate(self.evaluation_history):
            overall_score = sum(evaluation.evaluation_scores.values()) / len(evaluation.evaluation_scores)
            logger.info(f"Iteration {i+1}: Overall Score = {overall_score:.3f}")
            
            # Log individual dimension scores
            for dimension, score in evaluation.evaluation_scores.items():
                logger.info(f"  {dimension}: {score:.3f}")
        
        # Log best performing iteration
        best_iteration = max(
            range(len(self.evaluation_history)),
            key=lambda i: sum(self.evaluation_history[i].evaluation_scores.values())
        )
        logger.info(f"Best performing iteration: {best_iteration + 1}")
        
    def get_evaluation_results(self) -> List[EvaluationResult]:
        """Get the complete evaluation history"""
        return self.evaluation_history
    
    def get_best_iteration(self) -> Optional[EvaluationResult]:
        """Get the best performing iteration based on overall score"""
        if not self.evaluation_history:
            return None
            
                 return max(
             self.evaluation_history,
             key=lambda eval_result: sum(eval_result.evaluation_scores.values())
         )

    def create_agents(self, task: str):
        """Create agents for a given task.

        Args:
            task (str): The task to create agents for

        Returns:
            List[Agent]: List of created agents

        Raises:
            Exception: If there's an error during agent creation
        """
        try:
            logger.info(f"Creating agents for task: {task}")
            model = OpenAIFunctionCaller(
                system_prompt=BOSS_SYSTEM_PROMPT,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.5,
                base_model=AgentsConfig,
            )

            logger.info(
                "Getting agent configurations from boss agent"
            )
            output = model.run(
                f"Create the agents for the following task: {task}"
            )
            logger.debug(
                f"Received agent configurations: {output.model_dump()}"
            )
            output = output.model_dump()

            agents = []
            if isinstance(output, dict):
                for agent_config in output["agents"]:
                    logger.info(
                        f"Building agent: {agent_config['name']}"
                    )
                    agent = self.build_agent(
                        agent_name=agent_config["name"],
                        agent_description=agent_config["description"],
                        agent_system_prompt=agent_config[
                            "system_prompt"
                        ],
                    )
                    agents.append(agent)
                    logger.info(
                        f"Successfully built agent: {agent_config['name']}"
                    )

            return agents
        except Exception as e:
            logger.error(
                f"Error creating agents: {str(e)}", exc_info=True
            )
            raise

    def build_agent(
        self,
        agent_name: str,
        agent_description: str,
        agent_system_prompt: str,
    ) -> Agent:
        """Build a single agent with enhanced error handling.

        Args:
            agent_name (str): Name of the agent
            agent_description (str): Description of the agent
            agent_system_prompt (str): System prompt for the agent

        Returns:
            Agent: The constructed agent

        Raises:
            Exception: If there's an error during agent construction
        """
        logger.info(f"Building agent: {agent_name}")
        try:
            agent = Agent(
                agent_name=agent_name,
                description=agent_description,
                system_prompt=agent_system_prompt,
                verbose=self.verbose,
                dynamic_temperature_enabled=False,
            )
            logger.info(f"Successfully built agent: {agent_name}")
            return agent
        except Exception as e:
            logger.error(
                f"Error building agent {agent_name}: {str(e)}",
                exc_info=True,
            )
            raise

    def initialize_swarm_router(self, agents: List[Agent], task: str):
        """Initialize and run the swarm router.

        Args:
            agents (List[Agent]): List of agents to use
            task (str): The task to execute

        Returns:
            Any: The result of the swarm router execution

        Raises:
            Exception: If there's an error during router initialization or execution
        """
        try:
            logger.info("Initializing swarm router")
            model = OpenAIFunctionCaller(
                system_prompt=BOSS_SYSTEM_PROMPT,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.5,
                base_model=SwarmRouterConfig,
            )

            logger.info("Creating swarm specification")
            swarm_spec = model.run(
                f"Create the swarm spec for the following task: {task}"
            )
            logger.debug(
                f"Received swarm specification: {swarm_spec.model_dump()}"
            )
            swarm_spec = swarm_spec.model_dump()

            logger.info("Initializing SwarmRouter")
            swarm_router = SwarmRouter(
                name=swarm_spec["name"],
                description=swarm_spec["description"],
                max_loops=1,
                swarm_type=swarm_spec["swarm_type"],
                rearrange_flow=swarm_spec["rearrange_flow"],
                rules=swarm_spec["rules"],
                multi_agent_collab_prompt=swarm_spec[
                    "multi_agent_collab_prompt"
                ],
                agents=agents,
                output_type="dict",
            )

            logger.info("Starting swarm router execution")
            return swarm_router.run(task)
        except Exception as e:
            logger.error(
                f"Error in swarm router initialization/execution: {str(e)}",
                exc_info=True,
            )
            raise

    def batch_run(self, tasks: List[str]):
        """Run the swarm on a list of tasks.

        Args:
            tasks (List[str]): List of tasks to execute

        Returns:
            List[Any]: List of results from each task execution

        Raises:
            Exception: If there's an error during batch execution
        """

        return [self.run(task) for task in tasks]
