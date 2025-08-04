# GraphWorkflow API Endpoint Design

## Overview

This document outlines the design for a single API endpoint that allows users to create, configure, and execute GraphWorkflow instances. The endpoint provides a comprehensive interface for leveraging the GraphWorkflow functionality with minimal setup.

## Base URL

```
POST /api/v1/graph-workflow/execute
```

## Request Schema

### Main Request Body

```json
{
  "workflow_config": {
    "name": "string",
    "description": "string",
    "max_loops": 1,
    "auto_compile": true,
    "verbose": false
  },
  "agents": [
    {
      "id": "string",
      "agent_name": "string",
      "model_name": "string",
      "system_prompt": "string",
      "temperature": 0.7,
      "max_tokens": 4000,
      "max_loops": 1,
      "metadata": {}
    }
  ],
  "connections": [
    {
      "type": "simple",
      "source": "string",
      "target": "string",
      "metadata": {}
    }
  ],
  "entry_points": ["string"],
  "end_points": ["string"],
  "task": "string",
  "options": {
    "include_conversation": false,
    "include_runtime_state": false,
    "visualization": {
      "enabled": false,
      "format": "png",
      "show_summary": true
    }
  }
}
```

### Detailed Schema Definitions

#### WorkflowConfig
```json
{
  "name": "Investment Analysis Workflow",
  "description": "Multi-agent workflow for comprehensive investment analysis",
  "max_loops": 1,
  "auto_compile": true,
  "verbose": false
}
```

#### Agent Definition
```json
{
  "id": "fundamental_analyst",
  "agent_name": "Fundamental Analysis Agent",
  "model_name": "gpt-4o-mini",
  "system_prompt": "You are a fundamental analysis expert specializing in financial analysis...",
  "temperature": 0.7,
  "max_tokens": 4000,
  "max_loops": 1,
  "autosave": true,
  "dashboard": false,
  "metadata": {
    "specialization": "financial_analysis",
    "expertise_level": "expert"
  }
}
```

#### Connection Types

##### Simple Connection
```json
{
  "type": "simple",
  "source": "data_gatherer",
  "target": "fundamental_analyst",
  "metadata": {
    "priority": "high"
  }
}
```

##### Fan-out Connection
```json
{
  "type": "fan_out",
  "source": "data_gatherer",
  "targets": ["fundamental_analyst", "technical_analyst", "sentiment_analyst"],
  "metadata": {
    "parallel_execution": true
  }
}
```

##### Fan-in Connection
```json
{
  "type": "fan_in",
  "sources": ["fundamental_analyst", "technical_analyst", "sentiment_analyst"],
  "target": "synthesis_agent",
  "metadata": {
    "aggregation_method": "combine_all"
  }
}
```

##### Parallel Chain
```json
{
  "type": "parallel_chain",
  "sources": ["data_gatherer_1", "data_gatherer_2"],
  "targets": ["analyst_1", "analyst_2", "analyst_3"],
  "metadata": {
    "full_mesh": true
  }
}
```

## Response Schema

### Success Response
```json
{
  "status": "success",
  "workflow_id": "uuid-string",
  "execution_time": 45.23,
  "results": {
    "fundamental_analyst": "Analysis output from fundamental analyst...",
    "technical_analyst": "Technical analysis results...",
    "synthesis_agent": "Combined analysis and recommendations..."
  },
  "conversation": {
    "history": [
      {
        "role": "fundamental_analyst",
        "content": "Analysis output...",
        "timestamp": "2024-01-15T10:30:00Z"
      }
    ]
  },
  "metrics": {
    "total_agents": 5,
    "total_connections": 6,
    "execution_layers": 3,
    "parallel_efficiency": 85.5
  },
  "visualization": {
    "url": "https://api.example.com/visualizations/workflow_123.png",
    "format": "png"
  },
  "workflow_summary": {
    "name": "Investment Analysis Workflow",
    "description": "Multi-agent workflow for comprehensive investment analysis",
    "entry_points": ["data_gatherer"],
    "end_points": ["synthesis_agent"],
    "compilation_status": "compiled"
  }
}
```

### Error Response
```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Invalid workflow configuration",
  "details": {
    "field": "connections",
    "issue": "Source node 'invalid_node' does not exist",
    "suggestions": ["Check node IDs in connections", "Verify all referenced nodes exist"]
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Implementation Example

### Python FastAPI Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
import time
from swarms import Agent, GraphWorkflow, Node, NodeType, Edge

app = FastAPI(title="GraphWorkflow API", version="1.0.0")

# Pydantic Models
class WorkflowConfig(BaseModel):
    name: str = "Graph-Workflow-01"
    description: str = "A customizable workflow system"
    max_loops: int = 1
    auto_compile: bool = True
    verbose: bool = False

class AgentDefinition(BaseModel):
    id: str
    agent_name: str
    model_name: str = "gpt-4o-mini"
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4000
    max_loops: int = 1
    autosave: bool = True
    dashboard: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SimpleConnection(BaseModel):
    type: str = "simple"
    source: str
    target: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FanOutConnection(BaseModel):
    type: str = "fan_out"
    source: str
    targets: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FanInConnection(BaseModel):
    type: str = "fan_in"
    sources: List[str]
    target: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ParallelChainConnection(BaseModel):
    type: str = "parallel_chain"
    sources: List[str]
    targets: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VisualizationOptions(BaseModel):
    enabled: bool = False
    format: str = "png"
    show_summary: bool = True

class WorkflowOptions(BaseModel):
    include_conversation: bool = False
    include_runtime_state: bool = False
    visualization: VisualizationOptions = Field(default_factory=VisualizationOptions)

class GraphWorkflowRequest(BaseModel):
    workflow_config: WorkflowConfig
    agents: List[AgentDefinition]
    connections: List[Dict[str, Any]]  # Union of all connection types
    entry_points: Optional[List[str]] = None
    end_points: Optional[List[str]] = None
    task: str
    options: WorkflowOptions = Field(default_factory=WorkflowOptions)

@app.post("/api/v1/graph-workflow/execute")
async def execute_graph_workflow(request: GraphWorkflowRequest):
    """
    Execute a GraphWorkflow with the provided configuration.
    
    This endpoint creates a workflow from the provided agents and connections,
    executes it with the given task, and returns the results.
    """
    start_time = time.time()
    workflow_id = str(uuid.uuid4())
    
    try:
        # Create agents from definitions
        agent_instances = {}
        for agent_def in request.agents:
            agent = Agent(
                agent_name=agent_def.agent_name,
                model_name=agent_def.model_name,
                system_prompt=agent_def.system_prompt,
                temperature=agent_def.temperature,
                max_tokens=agent_def.max_tokens,
                max_loops=agent_def.max_loops,
                autosave=agent_def.autosave,
                dashboard=agent_def.dashboard,
            )
            agent_instances[agent_def.id] = agent
        
        # Create workflow
        workflow = GraphWorkflow(
            id=workflow_id,
            name=request.workflow_config.name,
            description=request.workflow_config.description,
            max_loops=request.workflow_config.max_loops,
            auto_compile=request.workflow_config.auto_compile,
            verbose=request.workflow_config.verbose,
        )
        
        # Add agents to workflow
        for agent_def in request.agents:
            workflow.add_node(agent_instances[agent_def.id])
        
        # Add connections
        for connection in request.connections:
            conn_type = connection.get("type", "simple")
            
            if conn_type == "simple":
                workflow.add_edge(connection["source"], connection["target"])
            elif conn_type == "fan_out":
                workflow.add_edges_from_source(
                    connection["source"], 
                    connection["targets"]
                )
            elif conn_type == "fan_in":
                workflow.add_edges_to_target(
                    connection["sources"], 
                    connection["target"]
                )
            elif conn_type == "parallel_chain":
                workflow.add_parallel_chain(
                    connection["sources"], 
                    connection["targets"]
                )
        
        # Set entry and end points
        if request.entry_points:
            workflow.set_entry_points(request.entry_points)
        else:
            workflow.auto_set_entry_points()
            
        if request.end_points:
            workflow.set_end_points(request.end_points)
        else:
            workflow.auto_set_end_points()
        
        # Execute workflow
        results = workflow.run(request.task)
        
        # Prepare response
        execution_time = time.time() - start_time
        
        response = {
            "status": "success",
            "workflow_id": workflow_id,
            "execution_time": execution_time,
            "results": results,
            "metrics": {
                "total_agents": len(workflow.nodes),
                "total_connections": len(workflow.edges),
                "execution_layers": len(workflow._sorted_layers) if workflow._compiled else 0,
                "parallel_efficiency": calculate_parallel_efficiency(workflow)
            },
            "workflow_summary": {
                "name": workflow.name,
                "description": workflow.description,
                "entry_points": workflow.entry_points,
                "end_points": workflow.end_points,
                "compilation_status": "compiled" if workflow._compiled else "not_compiled"
            }
        }
        
        # Add conversation if requested
        if request.options.include_conversation and workflow.conversation:
            response["conversation"] = {
                "history": workflow.conversation.history
            }
        
        # Add visualization if requested
        if request.options.visualization.enabled:
            try:
                viz_path = workflow.visualize(
                    format=request.options.visualization.format,
                    view=False,
                    show_summary=request.options.visualization.show_summary
                )
                response["visualization"] = {
                    "url": f"/api/v1/visualizations/{workflow_id}.{request.options.visualization.format}",
                    "format": request.options.visualization.format,
                    "local_path": viz_path
                }
            except Exception as e:
                response["visualization"] = {
                    "error": str(e),
                    "fallback": workflow.visualize_simple()
                }
        
        return response
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "error_code": "EXECUTION_ERROR",
                "message": str(e),
                "execution_time": execution_time,
                "workflow_id": workflow_id
            }
        )

def calculate_parallel_efficiency(workflow):
    """Calculate parallel execution efficiency percentage."""
    if not workflow._compiled or not workflow._sorted_layers:
        return 0.0
    
    total_nodes = len(workflow.nodes)
    max_parallel = max(len(layer) for layer in workflow._sorted_layers)
    
    if total_nodes == 0:
        return 0.0
    
    return (max_parallel / total_nodes) * 100

# Additional endpoints for workflow management
@app.get("/api/v1/graph-workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get the status of a workflow execution."""
    # Implementation for retrieving workflow status
    pass

@app.delete("/api/v1/graph-workflow/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow and its associated resources."""
    # Implementation for cleaning up workflow resources
    pass
```

## Usage Examples

### Basic Investment Analysis Workflow

```json
{
  "workflow_config": {
    "name": "Investment Analysis Workflow",
    "description": "Multi-agent workflow for comprehensive investment analysis",
    "max_loops": 1,
    "verbose": false
  },
  "agents": [
    {
      "id": "data_gatherer",
      "agent_name": "Data Gathering Agent",
      "model_name": "gpt-4o-mini",
      "system_prompt": "You are a financial data gathering specialist. Collect relevant financial data, news, and market information.",
      "temperature": 0.3
    },
    {
      "id": "fundamental_analyst",
      "agent_name": "Fundamental Analysis Agent",
      "model_name": "gpt-4o-mini",
      "system_prompt": "You are a fundamental analysis expert. Analyze company financials, business model, and competitive position.",
      "temperature": 0.5
    },
    {
      "id": "technical_analyst",
      "agent_name": "Technical Analysis Agent",
      "model_name": "gpt-4o-mini",
      "system_prompt": "You are a technical analysis specialist. Analyze price charts, trends, and trading patterns.",
      "temperature": 0.5
    },
    {
      "id": "synthesis_agent",
      "agent_name": "Synthesis Agent",
      "model_name": "gpt-4o-mini",
      "system_prompt": "You are a synthesis expert. Combine all analysis outputs into comprehensive investment recommendations.",
      "temperature": 0.7
    }
  ],
  "connections": [
    {
      "type": "fan_out",
      "source": "data_gatherer",
      "targets": ["fundamental_analyst", "technical_analyst"]
    },
    {
      "type": "fan_in",
      "sources": ["fundamental_analyst", "technical_analyst"],
      "target": "synthesis_agent"
    }
  ],
  "task": "Analyze the investment potential of Tesla (TSLA) stock based on current market conditions, financial performance, and technical indicators. Provide a comprehensive recommendation with risk assessment.",
  "options": {
    "include_conversation": true,
    "visualization": {
      "enabled": true,
      "format": "png",
      "show_summary": true
    }
  }
}
```

### Content Creation Workflow

```json
{
  "workflow_config": {
    "name": "Content Creation Workflow",
    "description": "Multi-stage content creation with research, writing, and review",
    "max_loops": 1
  },
  "agents": [
    {
      "id": "researcher",
      "agent_name": "Research Agent",
      "model_name": "gpt-4o-mini",
      "system_prompt": "You are a research specialist. Gather comprehensive information on the given topic.",
      "temperature": 0.3
    },
    {
      "id": "writer",
      "agent_name": "Content Writer",
      "model_name": "gpt-4o-mini",
      "system_prompt": "You are a professional content writer. Create engaging, well-structured content based on research.",
      "temperature": 0.7
    },
    {
      "id": "editor",
      "agent_name": "Editor",
      "model_name": "gpt-4o-mini",
      "system_prompt": "You are an expert editor. Review and improve content for clarity, accuracy, and engagement.",
      "temperature": 0.5
    }
  ],
  "connections": [
    {
      "type": "simple",
      "source": "researcher",
      "target": "writer"
    },
    {
      "type": "simple",
      "source": "writer",
      "target": "editor"
    }
  ],
  "task": "Create a comprehensive blog post about the future of artificial intelligence in healthcare, including current applications, challenges, and future prospects.",
  "options": {
    "include_conversation": true
  }
}
```

## Error Handling

### Common Error Codes

- `VALIDATION_ERROR`: Invalid workflow configuration
- `AGENT_CREATION_ERROR`: Failed to create agent instances
- `CONNECTION_ERROR`: Invalid connections between agents
- `EXECUTION_ERROR`: Workflow execution failed
- `VISUALIZATION_ERROR`: Failed to generate visualization
- `TIMEOUT_ERROR`: Workflow execution timed out

### Error Response Format

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "message": "Invalid workflow configuration",
  "details": {
    "field": "connections",
    "issue": "Source node 'invalid_node' does not exist",
    "suggestions": [
      "Check node IDs in connections",
      "Verify all referenced nodes exist"
    ]
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "workflow_id": "uuid-string"
}
```

## Rate Limiting and Quotas

- **Rate Limit**: 10 requests per minute per API key
- **Timeout**: 300 seconds (5 minutes) for workflow execution
- **Max Agents**: 50 agents per workflow
- **Max Connections**: 200 connections per workflow
- **Payload Size**: 10MB maximum request size

## Authentication

The API requires authentication using API keys:

```
Authorization: Bearer your-api-key-here
```

## Monitoring and Logging

- All workflow executions are logged with execution time and results
- Failed executions are logged with detailed error information
- Performance metrics are collected for optimization
- Workflow visualizations are cached for 24 hours

## Best Practices

1. **Agent Design**: Use clear, specific system prompts for each agent
2. **Connection Patterns**: Leverage fan-out and fan-in patterns for parallel processing
3. **Task Definition**: Provide clear, specific tasks for better results
4. **Error Handling**: Always check the response status and handle errors appropriately
5. **Resource Management**: Clean up workflows when no longer needed
6. **Testing**: Test workflows with smaller datasets before scaling up

## Future Enhancements

- **Streaming Responses**: Real-time workflow execution updates
- **Workflow Templates**: Pre-built workflow configurations
- **Scheduling**: Automated workflow execution on schedules
- **Versioning**: Workflow version control and rollback
- **Collaboration**: Multi-user workflow editing and sharing
- **Advanced Analytics**: Detailed performance and efficiency metrics 