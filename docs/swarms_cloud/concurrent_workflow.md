# ConcurrentWorkflow

*Runs independent tasks in parallel for faster processing*

**Swarm Type**: `ConcurrentWorkflow`

## Overview

The ConcurrentWorkflow swarm type runs independent tasks in parallel, significantly reducing processing time for complex operations. This architecture is ideal for tasks that can be processed simultaneously without dependencies, allowing multiple agents to work on different aspects of a problem at the same time.

Key features:
- **Parallel Execution**: Multiple agents work simultaneously
- **Reduced Processing Time**: Faster completion through parallelization
- **Independent Tasks**: Agents work on separate, non-dependent subtasks
- **Scalable Performance**: Performance scales with the number of agents

## Use Cases

- Independent data analysis tasks
- Parallel content generation
- Multi-source research projects
- Distributed problem solving

## API Usage

### Basic ConcurrentWorkflow Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Market Research Concurrent",
        "description": "Parallel market research across different sectors",
        "swarm_type": "ConcurrentWorkflow",
        "task": "Research and analyze market opportunities in AI, healthcare, fintech, and e-commerce sectors",
        "agents": [
          {
            "agent_name": "AI Market Analyst",
            "description": "Analyzes AI market trends and opportunities",
            "system_prompt": "You are an AI market analyst. Focus on artificial intelligence market trends, opportunities, key players, and growth projections.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Healthcare Market Analyst",
            "description": "Analyzes healthcare market trends",
            "system_prompt": "You are a healthcare market analyst. Focus on healthcare market trends, digital health opportunities, regulatory landscape, and growth areas.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Fintech Market Analyst",
            "description": "Analyzes fintech market opportunities",
            "system_prompt": "You are a fintech market analyst. Focus on financial technology trends, digital payment systems, blockchain opportunities, and regulatory developments.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "E-commerce Market Analyst",
            "description": "Analyzes e-commerce market trends",
            "system_prompt": "You are an e-commerce market analyst. Focus on online retail trends, marketplace opportunities, consumer behavior, and emerging platforms.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          }
        ],
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
        "name": "Market Research Concurrent",
        "description": "Parallel market research across different sectors",
        "swarm_type": "ConcurrentWorkflow",
        "task": "Research and analyze market opportunities in AI, healthcare, fintech, and e-commerce sectors",
        "agents": [
            {
                "agent_name": "AI Market Analyst",
                "description": "Analyzes AI market trends and opportunities",
                "system_prompt": "You are an AI market analyst. Focus on artificial intelligence market trends, opportunities, key players, and growth projections.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Healthcare Market Analyst",
                "description": "Analyzes healthcare market trends",
                "system_prompt": "You are a healthcare market analyst. Focus on healthcare market trends, digital health opportunities, regulatory landscape, and growth areas.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Fintech Market Analyst",
                "description": "Analyzes fintech market opportunities",
                "system_prompt": "You are a fintech market analyst. Focus on financial technology trends, digital payment systems, blockchain opportunities, and regulatory developments.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "E-commerce Market Analyst",
                "description": "Analyzes e-commerce market trends",
                "system_prompt": "You are an e-commerce market analyst. Focus on online retail trends, marketplace opportunities, consumer behavior, and emerging platforms.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            }
        ],
        "max_loops": 1
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print("ConcurrentWorkflow swarm completed successfully!")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Parallel results: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
    "job_id": "swarms-S17nZFDesmLHxCRoeyF3NVYvPaXk",
    "status": "success",
    "swarm_name": "Market Research Concurrent",
    "description": "Parallel market research across different sectors",
    "swarm_type": "ConcurrentWorkflow",
    "output": [
        {
            "role": "E-commerce Market Analyst",
            "content": "To analyze market opportunities in the AI, healthcare, fintech, and e-commerce sectors, we can break down each sector's current trends, consumer behavior, and emerging platforms. Here's an overview of each sector with a focus on e-commerce....."
        },
        {
            "role": "AI Market Analyst",
            "content": "The artificial intelligence (AI) landscape presents numerous opportunities across various sectors, particularly in healthcare, fintech, and e-commerce. Here's a detailed analysis of each sector:\n\n### Healthcare....."
        },
        {
            "role": "Healthcare Market Analyst",
            "content": "As a Healthcare Market Analyst, I will focus on analyzing market opportunities within the healthcare sector, particularly in the realm of AI and digital health. The intersection of healthcare with fintech and e-commerce also presents unique opportunities. Here's an overview of key trends and growth areas:...."
        },
        {
            "role": "Fintech Market Analyst",
            "content": "Certainly! Let's break down the market opportunities in the fintech sector, focusing on financial technology trends, digital payment systems, blockchain opportunities, and regulatory developments:\n\n### 1. Financial Technology Trends....."
        }
    ],
    "number_of_agents": 4,
    "service_tier": "standard",
    "execution_time": 23.360230922698975,
    "usage": {
        "input_tokens": 35,
        "output_tokens": 2787,
        "total_tokens": 2822,
        "billing_info": {
            "cost_breakdown": {
                "agent_cost": 0.04,
                "input_token_cost": 0.000105,
                "output_token_cost": 0.041805,
                "token_counts": {
                    "total_input_tokens": 35,
                    "total_output_tokens": 2787,
                    "total_tokens": 2822
                },
                "num_agents": 4,
                "service_tier": "standard",
                "night_time_discount_applied": true
            },
            "total_cost": 0.08191,
            "discount_active": true,
            "discount_type": "night_time",
            "discount_percentage": 75
        }
    }
}
```

## Best Practices

- Design independent tasks that don't require sequential dependencies
- Use for tasks that can be parallelized effectively
- Ensure agents have distinct, non-overlapping responsibilities
- Ideal for time-sensitive analysis requiring multiple perspectives

## Related Swarm Types

- [SequentialWorkflow](sequential_workflow.md) - For ordered execution
- [MixtureOfAgents](mixture_of_agents.md) - For collaborative analysis
- [MultiAgentRouter](multi_agent_router.md) - For intelligent task distribution