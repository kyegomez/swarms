from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class AgentStep(BaseModel):
    """Schema for individual agent execution steps"""
    role: str = Field(description="Role of the agent in this step")
    content: str = Field(description="Content/response from the agent")

class AgentOutput(BaseModel):
    """Schema for individual agent outputs"""
    agent_name: str = Field(description="Name of the agent")
    steps: List[AgentStep] = Field(description="List of execution steps by this agent")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the agent's execution")

class SwarmInput(BaseModel):
    """Schema for swarm input configuration"""
    swarm_id: str = Field(description="Unique identifier for the swarm execution")
    name: str = Field(description="Name of the swarm type")
    flow: str = Field(description="Agent execution flow configuration")
    description: Optional[str] = Field(default=None, description="Description of the swarm execution")

class SwarmOutput(BaseModel):
    """Unified schema for all swarm type outputs"""
    input: SwarmInput = Field(description="Input configuration for the swarm")
    outputs: List[AgentOutput] = Field(description="List of outputs from all agents")
    time: float = Field(description="Timestamp of execution")
    execution_time: Optional[float] = Field(default=None, description="Total execution time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional swarm execution metadata")
    
    def format_output(self) -> str:
        """Format the swarm output into a readable string"""
        output = f"Workflow Execution Details\n\n"
        output += f"Swarm ID: `{self.input.swarm_id}`\n"
        output += f"Swarm Name: `{self.input.name}`\n"
        output += f"Agent Flow: `{self.input.flow}`\n\n---\n"
        output += f"Agent Task Execution\n\n"

        for i, agent_output in enumerate(self.outputs, start=1):
            output += f"Run {i} (Agent: `{agent_output.agent_name}`)\n\n"
            
            for j, step in enumerate(agent_output.steps, start=1):
                if step.role.strip() != "System:":
                    output += f"Step {j}:\n"
                    output += f"Response: {step.content}\n\n"

        if self.execution_time:
            output += f"Overall Execution Time: `{self.execution_time:.2f}s`"
        
        return output

class MixtureOfAgentsOutput(SwarmOutput):
    """Schema specific to MixtureOfAgents output"""
    aggregator_summary: Optional[str] = Field(default=None, description="Aggregated summary from all agents")
    
    def format_output(self) -> str:
        """Format MixtureOfAgents output"""
        output = super().format_output()
        if self.aggregator_summary:
            output += f"\nAggregated Summary:\n{self.aggregator_summary}\n{'=' * 50}\n"
        return output

class SpreadsheetSwarmOutput(SwarmOutput):
    """Schema specific to SpreadsheetSwarm output"""
    csv_data: List[List[str]] = Field(description="CSV data in list format")
    
    def format_output(self) -> str:
        """Format SpreadsheetSwarm output"""
        output = "### Spreadsheet Swarm Output ###\n\n"
        if self.csv_data:
            # Create markdown table
            header = self.csv_data[0]
            output += "| " + " | ".join(header) + " |\n"
            output += "| " + " | ".join(["---"] * len(header)) + " |\n"
            
            for row in self.csv_data[1:]:
                output += "| " + " | ".join(row) + " |\n"
        
        return output 