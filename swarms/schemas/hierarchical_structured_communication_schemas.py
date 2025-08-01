"""
Schemas for Talk Structurally, Act Hierarchically Framework

This module defines the Pydantic schemas used in the TalkHierarchical framework
for structured communication and hierarchical evaluation.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


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


class StructuredMessageSchema(BaseModel):
    """Schema for structured communication messages"""
    message: str = Field(
        description="Specific task instructions (M_ij)",
        min_length=1
    )
    background: str = Field(
        description="Context and problem background (B_ij)",
        default=""
    )
    intermediate_output: str = Field(
        description="Intermediate results (I_ij)",
        default=""
    )
    sender: str = Field(
        description="Name of the sending agent",
        min_length=1
    )
    recipient: str = Field(
        description="Name of the receiving agent",
        min_length=1
    )
    timestamp: Optional[str] = Field(
        description="Timestamp of the message",
        default=None
    )
    communication_type: CommunicationType = Field(
        description="Type of communication",
        default=CommunicationType.MESSAGE
    )


class HierarchicalOrderSchema(BaseModel):
    """Schema for hierarchical task orders"""
    agent_name: str = Field(
        description="Name of the agent to receive the task",
        min_length=1
    )
    task: str = Field(
        description="Specific task description",
        min_length=1
    )
    communication_type: CommunicationType = Field(
        description="Type of communication to use",
        default=CommunicationType.MESSAGE
    )
    background_context: str = Field(
        description="Background context for the task",
        default=""
    )
    intermediate_output: str = Field(
        description="Intermediate output to pass along",
        default=""
    )
    priority: int = Field(
        description="Task priority (1-10, higher is more important)",
        default=5,
        ge=1,
        le=10
    )


class EvaluationCriterionSchema(BaseModel):
    """Schema for evaluation criteria"""
    name: str = Field(
        description="Name of the evaluation criterion",
        min_length=1
    )
    description: str = Field(
        description="Description of what this criterion evaluates",
        min_length=1
    )
    weight: float = Field(
        description="Weight of this criterion in overall evaluation (0-1)",
        default=1.0,
        ge=0.0,
        le=1.0
    )
    scale: str = Field(
        description="Scale for evaluation (e.g., '0-10', 'A-F', 'Poor-Excellent')",
        default="0-10"
    )


class EvaluationResultSchema(BaseModel):
    """Schema for evaluation results"""
    evaluator_name: str = Field(
        description="Name of the evaluator",
        min_length=1
    )
    criterion: str = Field(
        description="Evaluation criterion",
        min_length=1
    )
    score: float = Field(
        description="Evaluation score",
        ge=0.0
    )
    feedback: str = Field(
        description="Detailed feedback",
        min_length=1
    )
    confidence: float = Field(
        description="Confidence in evaluation (0-1)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Reasoning behind the evaluation",
        default=""
    )
    suggestions: List[str] = Field(
        description="Suggestions for improvement",
        default=[]
    )


class EvaluationSummarySchema(BaseModel):
    """Schema for evaluation summaries"""
    overall_score: float = Field(
        description="Overall evaluation score",
        ge=0.0
    )
    criterion_scores: Dict[str, float] = Field(
        description="Scores for each criterion",
        default={}
    )
    strengths: List[str] = Field(
        description="Identified strengths",
        default=[]
    )
    weaknesses: List[str] = Field(
        description="Identified weaknesses",
        default=[]
    )
    recommendations: List[str] = Field(
        description="Recommendations for improvement",
        default=[]
    )
    confidence: float = Field(
        description="Overall confidence in evaluation",
        ge=0.0,
        le=1.0
    )


class AgentConfigSchema(BaseModel):
    """Schema for agent configuration"""
    name: str = Field(
        description="Name of the agent",
        min_length=1
    )
    role: AgentRole = Field(
        description="Role of the agent in the system"
    )
    model_name: str = Field(
        description="Model to use for this agent",
        default="gpt-4o-mini"
    )
    system_prompt: str = Field(
        description="System prompt for the agent",
        min_length=1
    )
    tools: List[str] = Field(
        description="Tools available to this agent",
        default=[]
    )
    capabilities: List[str] = Field(
        description="Capabilities of this agent",
        default=[]
    )
    evaluation_criteria: List[str] = Field(
        description="Evaluation criteria this agent can assess",
        default=[]
    )


class TalkHierarchicalConfigSchema(BaseModel):
    """Schema for TalkHierarchical swarm configuration"""
    name: str = Field(
        description="Name of the swarm",
        default="TalkHierarchicalSwarm"
    )
    description: str = Field(
        description="Description of the swarm",
        default="Talk Structurally, Act Hierarchically Framework"
    )
    max_loops: int = Field(
        description="Maximum number of refinement loops",
        default=3,
        ge=1
    )
    enable_structured_communication: bool = Field(
        description="Enable structured communication protocol",
        default=True
    )
    enable_hierarchical_evaluation: bool = Field(
        description="Enable hierarchical evaluation system",
        default=True
    )
    shared_memory: bool = Field(
        description="Enable shared memory between agents",
        default=True
    )
    evaluation_criteria: List[EvaluationCriterionSchema] = Field(
        description="Evaluation criteria for the system",
        default=[]
    )
    agents: List[AgentConfigSchema] = Field(
        description="Configuration for all agents",
        default=[]
    )
    quality_threshold: float = Field(
        description="Quality threshold for stopping refinement",
        default=8.0,
        ge=0.0,
        le=10.0
    )


class TalkHierarchicalStateSchema(BaseModel):
    """Schema for TalkHierarchical swarm state"""
    conversation_history: List[StructuredMessageSchema] = Field(
        description="History of structured messages",
        default=[]
    )
    intermediate_outputs: Dict[str, str] = Field(
        description="Intermediate outputs from agents",
        default={}
    )
    evaluation_results: List[EvaluationResultSchema] = Field(
        description="Results from evaluations",
        default=[]
    )
    current_loop: int = Field(
        description="Current refinement loop",
        default=0
    )
    task: str = Field(
        description="Current task being processed",
        default=""
    )
    final_result: Optional[str] = Field(
        description="Final result of the workflow",
        default=None
    )


class TalkHierarchicalResponseSchema(BaseModel):
    """Schema for TalkHierarchical swarm response"""
    final_result: str = Field(
        description="Final result of the workflow"
    )
    total_loops: int = Field(
        description="Total number of loops executed",
        ge=1
    )
    conversation_history: List[StructuredMessageSchema] = Field(
        description="Complete conversation history"
    )
    evaluation_results: List[EvaluationResultSchema] = Field(
        description="All evaluation results"
    )
    intermediate_outputs: Dict[str, str] = Field(
        description="All intermediate outputs"
    )
    evaluation_summary: Optional[EvaluationSummarySchema] = Field(
        description="Summary of evaluations",
        default=None
    )
    performance_metrics: Dict[str, Union[float, int, str]] = Field(
        description="Performance metrics",
        default={}
    )


class GraphNodeSchema(BaseModel):
    """Schema for graph nodes in agent orchestration"""
    node_id: str = Field(
        description="Unique identifier for the node",
        min_length=1
    )
    agent_name: str = Field(
        description="Name of the agent at this node",
        min_length=1
    )
    role: AgentRole = Field(
        description="Role of the agent"
    )
    capabilities: List[str] = Field(
        description="Capabilities of this node",
        default=[]
    )
    connections: List[str] = Field(
        description="IDs of connected nodes",
        default=[]
    )
    conditions: Dict[str, str] = Field(
        description="Conditions for routing to this node",
        default={}
    )


class GraphSchema(BaseModel):
    """Schema for agent orchestration graph"""
    nodes: List[GraphNodeSchema] = Field(
        description="Nodes in the graph",
        default=[]
    )
    start_node: str = Field(
        description="ID of the start node",
        min_length=1
    )
    end_nodes: List[str] = Field(
        description="IDs of end nodes",
        default=[]
    )
    routing_rules: Dict[str, str] = Field(
        description="Rules for routing between nodes",
        default={}
    )
    max_depth: int = Field(
        description="Maximum depth of the graph",
        default=10,
        ge=1
    )


# Response schemas for different agent types
class GeneratorResponseSchema(BaseModel):
    """Schema for generator agent responses"""
    content: str = Field(
        description="Generated content",
        min_length=1
    )
    intermediate_output: str = Field(
        description="Intermediate output for next agent",
        default=""
    )
    reasoning: str = Field(
        description="Reasoning behind the generation",
        default=""
    )
    confidence: float = Field(
        description="Confidence in the generated content",
        ge=0.0,
        le=1.0
    )


class EvaluatorResponseSchema(BaseModel):
    """Schema for evaluator agent responses"""
    criterion: str = Field(
        description="Evaluation criterion",
        min_length=1
    )
    score: float = Field(
        description="Evaluation score",
        ge=0.0
    )
    feedback: str = Field(
        description="Detailed feedback",
        min_length=1
    )
    confidence: float = Field(
        description="Confidence in evaluation",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Reasoning behind the evaluation",
        default=""
    )
    suggestions: List[str] = Field(
        description="Suggestions for improvement",
        default=[]
    )


class RefinerResponseSchema(BaseModel):
    """Schema for refiner agent responses"""
    refined_content: str = Field(
        description="Refined content",
        min_length=1
    )
    changes_made: List[str] = Field(
        description="List of changes made",
        default=[]
    )
    reasoning: str = Field(
        description="Reasoning behind refinements",
        default=""
    )
    confidence: float = Field(
        description="Confidence in refinements",
        ge=0.0,
        le=1.0
    )
    feedback_addressed: List[str] = Field(
        description="Feedback points addressed",
        default=[]
    ) 