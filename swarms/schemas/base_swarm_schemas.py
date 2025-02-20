from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import uuid
import time

class AgentInputConfig(BaseModel):
    """
    Configuration for an agent. This can be further customized
    per agent type if needed.
    """
    agent_name: str = Field(..., description="Name of the agent")
    agent_type: str = Field(..., description="Type of agent (e.g. 'llm', 'tool', 'memory')")
    model_name: Optional[str] = Field(None, description="Name of the model to use")
    temperature: float = Field(0.7, description="Temperature for model sampling")
    max_tokens: int = Field(4096, description="Maximum tokens for model response")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    tools: Optional[List[str]] = Field(None, description="List of tool names available to agent")
    memory_type: Optional[str] = Field(None, description="Type of memory to use")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional agent metadata")

class BaseSwarmSchema(BaseModel):
    """
    Base schema for all swarm types.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    agents: List[AgentInputConfig]  # Using AgentInputConfig
    max_loops: int = 1
    swarm_type: str  # e.g., "SequentialWorkflow", "ConcurrentWorkflow", etc.
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    config: Dict[str, Any] = Field(default_factory=dict)  # Flexible config
    
    # Additional fields
    timeout: Optional[int] = Field(None, description="Timeout in seconds for swarm execution")
    error_handling: str = Field("stop", description="Error handling strategy: 'stop', 'continue', or 'retry'")
    max_retries: int = Field(3, description="Maximum number of retry attempts")
    logging_level: str = Field("info", description="Logging level for the swarm")
    metrics_enabled: bool = Field(True, description="Whether to collect metrics")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing swarms")
    
    @validator("swarm_type")
    def validate_swarm_type(cls, v):
        """Validates the swarm type is one of the allowed types"""
        allowed_types = [
            "SequentialWorkflow",
            "ConcurrentWorkflow", 
            "AgentRearrange",
            "MixtureOfAgents",
            "SpreadSheetSwarm",
            "AutoSwarm",
            "HierarchicalSwarm",
            "FeedbackSwarm"
        ]
        if v not in allowed_types:
            raise ValueError(f"Swarm type must be one of: {allowed_types}")
        return v

    @validator("config")
    def validate_config(cls, v, values):
        """
        Validates the 'config' dictionary based on the 'swarm_type'.
        """
        swarm_type = values.get("swarm_type")
        
        # Common validation for all swarm types
        if not isinstance(v, dict):
            raise ValueError("Config must be a dictionary")

        # Type-specific validation
        if swarm_type == "SequentialWorkflow":
            if "flow" not in v:
                raise ValueError("SequentialWorkflow requires a 'flow' configuration.")
            if not isinstance(v["flow"], list):
                raise ValueError("Flow configuration must be a list")

        elif swarm_type == "ConcurrentWorkflow":
            if "max_workers" not in v:
                raise ValueError("ConcurrentWorkflow requires a 'max_workers' configuration.")
            if not isinstance(v["max_workers"], int) or v["max_workers"] < 1:
                raise ValueError("max_workers must be a positive integer")

        elif swarm_type == "AgentRearrange":
            if "flow" not in v:
                raise ValueError("AgentRearrange requires a 'flow' configuration.")
            if not isinstance(v["flow"], list):
                raise ValueError("Flow configuration must be a list")

        elif swarm_type == "MixtureOfAgents":
            if "aggregator_agent" not in v:
                raise ValueError("MixtureOfAgents requires an 'aggregator_agent' configuration.")
            if "voting_method" not in v:
                v["voting_method"] = "majority"  # Set default voting method

        elif swarm_type == "SpreadSheetSwarm":
            if "save_file_path" not in v:
                raise ValueError("SpreadSheetSwarm requires a 'save_file_path' configuration.")
            if not isinstance(v["save_file_path"], str):
                raise ValueError("save_file_path must be a string")

        elif swarm_type == "AutoSwarm":
            if "optimization_metric" not in v:
                v["optimization_metric"] = "performance"  # Set default metric
            if "adaptation_strategy" not in v:
                v["adaptation_strategy"] = "dynamic"  # Set default strategy

        elif swarm_type == "HierarchicalSwarm":
            if "hierarchy_levels" not in v:
                raise ValueError("HierarchicalSwarm requires 'hierarchy_levels' configuration.")
            if not isinstance(v["hierarchy_levels"], int) or v["hierarchy_levels"] < 1:
                raise ValueError("hierarchy_levels must be a positive integer")

        elif swarm_type == "FeedbackSwarm":
            if "feedback_collection" not in v:
                v["feedback_collection"] = "continuous"  # Set default collection method
            if "feedback_integration" not in v:
                v["feedback_integration"] = "weighted"  # Set default integration method

        return v

    @validator("error_handling")
    def validate_error_handling(cls, v):
        """Validates error handling strategy"""
        allowed_strategies = ["stop", "continue", "retry"]
        if v not in allowed_strategies:
            raise ValueError(f"Error handling must be one of: {allowed_strategies}")
        return v

    @validator("logging_level")
    def validate_logging_level(cls, v):
        """Validates logging level"""
        allowed_levels = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in allowed_levels:
            raise ValueError(f"Logging level must be one of: {allowed_levels}")
        return v.lower()

    def get_agent_by_name(self, name: str) -> Optional[AgentInputConfig]:
        """Helper method to get agent config by name"""
        for agent in self.agents:
            if agent.agent_name == name:
                return agent
        return None

    def add_tag(self, tag: str) -> None:
        """Helper method to add a tag"""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Helper method to remove a tag"""
        if tag in self.tags:
            self.tags.remove(tag)