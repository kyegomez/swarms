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
  "status": "success",
  "swarm_name": "market-research-concurrent",
  "swarm_type": "ConcurrentWorkflow",
  "task": "Research and analyze market opportunities in AI, healthcare, fintech, and e-commerce sectors",
  "output": {
    "ai_market_analysis": {
      "market_size": "$150B by 2025",
      "growth_rate": "25% CAGR",
      "key_opportunities": ["Generative AI", "Edge AI", "AI Infrastructure"]
    },
    "healthcare_analysis": {
      "market_size": "$350B by 2025",
      "growth_rate": "12% CAGR", 
      "key_opportunities": ["Telemedicine", "AI Diagnostics", "Digital Therapeutics"]
    },
    "fintech_analysis": {
      "market_size": "$200B by 2025",
      "growth_rate": "18% CAGR",
      "key_opportunities": ["DeFi", "Digital Banking", "Payment Infrastructure"]
    },
    "ecommerce_analysis": {
      "market_size": "$8T by 2025",
      "growth_rate": "14% CAGR",
      "key_opportunities": ["Social Commerce", "B2B Marketplaces", "Sustainable Commerce"]
    }
  },
  "metadata": {
    "parallel_execution": true,
    "agents_completed_simultaneously": 4,
    "execution_time_seconds": 12.8,
    "billing_info": {
      "total_cost": 0.052
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