# AgentRearrange

*Dynamically reorganizes agents to optimize task performance and efficiency*

**Swarm Type**: `AgentRearrange`

## Overview

The AgentRearrange swarm type dynamically reorganizes the workflow between agents based on task requirements and performance metrics. This architecture is particularly useful when the effectiveness of agents depends on their sequence or arrangement, allowing for optimal task distribution and execution flow.

Key features:
- **Dynamic Reorganization**: Automatically adjusts agent order based on task needs
- **Performance Optimization**: Optimizes workflow for maximum efficiency
- **Adaptive Sequencing**: Learns from execution patterns to improve arrangement
- **Flexible Task Distribution**: Distributes work based on agent capabilities

## Use Cases

- Complex workflows where task order matters
- Multi-step processes requiring optimization
- Tasks where agent performance varies by sequence
- Adaptive workflow management systems

## API Usage

### Basic AgentRearrange Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Document Processing Rearrange",
        "description": "Process documents with dynamic agent reorganization",
        "swarm_type": "AgentRearrange",
        "task": "Analyze this legal document and extract key insights, then summarize findings and identify action items",
        "agents": [
          {
            "agent_name": "Document Analyzer",
            "description": "Analyzes document content and structure",
            "system_prompt": "You are an expert document analyst. Extract key information, themes, and insights from documents.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Legal Expert",
            "description": "Provides legal context and interpretation",
            "system_prompt": "You are a legal expert. Analyze documents for legal implications, risks, and compliance issues.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.2
          },
          {
            "agent_name": "Summarizer",
            "description": "Creates concise summaries and action items",
            "system_prompt": "You are an expert at creating clear, actionable summaries from complex information.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.4
          }
        ],
        "rearrange_flow": "Summarizer -> Legal Expert -> Document Analyzer",
        "max_loops": 1
      }'
    ```

=== "Python (requests)"
    ```python
    import requests
    import json

    API_BASE_URL = "https://api.swarms.world"
    API_KEY = "your_api_key_here"
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    swarm_config = {
        "name": "Document Processing Rearrange",
        "description": "Process documents with dynamic agent reorganization",
        "swarm_type": "AgentRearrange",
        "task": "Analyze this legal document and extract key insights, then summarize findings and identify action items",
        "agents": [
            {
                "agent_name": "Document Analyzer",
                "description": "Analyzes document content and structure",
                "system_prompt": "You are an expert document analyst. Extract key information, themes, and insights from documents.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Legal Expert", 
                "description": "Provides legal context and interpretation",
                "system_prompt": "You are a legal expert. Analyze documents for legal implications, risks, and compliance issues.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.2
            },
            {
                "agent_name": "Summarizer",
                "description": "Creates concise summaries and action items", 
                "system_prompt": "You are an expert at creating clear, actionable summaries from complex information.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.4
            }
        ],
        "rearrange_flow": "Summarizer -> Legal Expert -> Document Analyzer",
        "max_loops": 1
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print("AgentRearrange swarm completed successfully!")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Output: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
    "job_id": "swarms-Uc8R7UcepLmNNPwcU7JC6YPy5wiI",
    "status": "success",
    "swarm_name": "Document Processing Rearrange",
    "description": "Process documents with dynamic agent reorganization",
    "swarm_type": "AgentRearrange",
    "output": [
        {
            "role": "Summarizer",
            "content": "\"Of course! Please provide the legal document you would like me to analyze, and I'll help extract key insights, summarize findings, and identify any action items.\""
        },
        {
            "role": "Legal Expert",
            "content": "\"\"Absolutely! Please upload or describe the legal document you need assistance with, and I'll provide an analysis that highlights key insights, summarizes the findings, and identifies any action items that may be necessary.\"\""
        },
        {
            "role": "Document Analyzer",
            "content": "\"Of course! Please provide the legal document you would like me to analyze, and I'll help extract key insights, summarize findings, and identify any action items.\""
        }
    ],
    "number_of_agents": 3,
    "service_tier": "standard",
    "execution_time": 7.898931264877319,
    "usage": {
        "input_tokens": 22,
        "output_tokens": 144,
        "total_tokens": 166,
        "billing_info": {
            "cost_breakdown": {
                "agent_cost": 0.03,
                "input_token_cost": 0.000066,
                "output_token_cost": 0.00216,
                "token_counts": {
                    "total_input_tokens": 22,
                    "total_output_tokens": 144,
                    "total_tokens": 166
                },
                "num_agents": 3,
                "service_tier": "standard",
                "night_time_discount_applied": true
            },
            "total_cost": 0.032226,
            "discount_active": true,
            "discount_type": "night_time",
            "discount_percentage": 75
        }
    }
}
```

## Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `rearrange_flow` | string | Instructions for how agents should be rearranged | None |
| `agents` | Array<AgentSpec> | List of agents to be dynamically arranged | Required |
| `max_loops` | integer | Maximum rearrangement iterations | 1 |

## Best Practices

- Provide clear `rearrange_flow` instructions for optimal reorganization
- Design agents with complementary but flexible roles
- Use when task complexity requires adaptive sequencing
- Monitor execution patterns to understand rearrangement decisions

## Related Swarm Types

- [SequentialWorkflow](sequential_workflow.md) - For fixed sequential processing
- [AutoSwarmBuilder](auto_swarm_builder.md) - For automatic swarm construction
- [HierarchicalSwarm](hierarchical_swarm.md) - For structured agent hierarchies