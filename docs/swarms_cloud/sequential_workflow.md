# SequentialWorkflow

*Executes tasks in a strict, predefined order for step-by-step processing*

**Swarm Type**: `SequentialWorkflow`

## Overview

The SequentialWorkflow swarm type executes tasks in a strict, predefined order where each step depends on the completion of the previous one. This architecture is perfect for workflows that require a linear progression of tasks, ensuring that each agent builds upon the work of the previous agent.

Key features:
- **Ordered Execution**: Agents execute in a specific, predefined sequence
- **Step Dependencies**: Each step builds upon previous results
- **Predictable Flow**: Clear, linear progression through the workflow
- **Quality Control**: Each agent can validate and enhance previous work

## Use Cases

- Document processing pipelines
- Multi-stage analysis workflows
- Content creation and editing processes
- Data transformation and validation pipelines

## API Usage

### Basic SequentialWorkflow Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Content Creation Pipeline",
        "description": "Sequential content creation from research to final output",
        "swarm_type": "SequentialWorkflow",
        "task": "Create a comprehensive blog post about the future of renewable energy",
        "agents": [
          {
            "agent_name": "Research Specialist",
            "description": "Conducts thorough research on the topic",
            "system_prompt": "You are a research specialist. Gather comprehensive, accurate information on the given topic from reliable sources.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Content Writer",
            "description": "Creates engaging written content",
            "system_prompt": "You are a skilled content writer. Transform research into engaging, well-structured articles that are informative and readable.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.6
          },
          {
            "agent_name": "Editor",
            "description": "Reviews and polishes the content",
            "system_prompt": "You are a professional editor. Review content for clarity, grammar, flow, and overall quality. Make improvements while maintaining the author's voice.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.4
          },
          {
            "agent_name": "SEO Optimizer",
            "description": "Optimizes content for search engines",
            "system_prompt": "You are an SEO expert. Optimize content for search engines while maintaining readability and quality.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.2
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
        "name": "Content Creation Pipeline",
        "description": "Sequential content creation from research to final output",
        "swarm_type": "SequentialWorkflow",
        "task": "Create a comprehensive blog post about the future of renewable energy",
        "agents": [
            {
                "agent_name": "Research Specialist",
                "description": "Conducts thorough research on the topic",
                "system_prompt": "You are a research specialist. Gather comprehensive, accurate information on the given topic from reliable sources.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Content Writer",
                "description": "Creates engaging written content",
                "system_prompt": "You are a skilled content writer. Transform research into engaging, well-structured articles that are informative and readable.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.6
            },
            {
                "agent_name": "Editor",
                "description": "Reviews and polishes the content",
                "system_prompt": "You are a professional editor. Review content for clarity, grammar, flow, and overall quality. Make improvements while maintaining the author's voice.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.4
            },
            {
                "agent_name": "SEO Optimizer",
                "description": "Optimizes content for search engines",
                "system_prompt": "You are an SEO expert. Optimize content for search engines while maintaining readability and quality.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.2
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
        print("SequentialWorkflow swarm completed successfully!")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Final output: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
    "job_id": "swarms-pbM8wqUwxq8afGeROV2A4xAcncd1",
    "status": "success",
    "swarm_name": "Content Creation Pipeline",
    "description": "Sequential content creation from research to final output",
    "swarm_type": "SequentialWorkflow",
    "output": [
        {
            "role": "Research Specialist",
            "content": "\"**Title: The Future of Renewable Energy: Charting a Sustainable Path Forward**\n\nAs we navigate the complexities of the 21st century, the transition to renewable energy stands out as a critical endeavor to ensure a sustainable future......"
        },
        {
            "role": "SEO Optimizer",
            "content": "\"**Title: The Future of Renewable Energy: Charting a Sustainable Path Forward**\n\nThe transition to renewable energy is crucial as we face the challenges of the 21st century, including climate change and dwindling fossil fuel resources......."
        },
        {
            "role": "Editor",
            "content": "\"**Title: The Future of Renewable Energy: Charting a Sustainable Path Forward**\n\nAs we confront the challenges of the 21st century, transitioning to renewable energy is essential for a sustainable future. With climate change concerns escalating and fossil fuel reserves depleting, renewable energy is not just an option but a necessity...."
        },
        {
            "role": "Content Writer",
            "content": "\"**Title: The Future of Renewable Energy: Charting a Sustainable Path Forward**\n\nAs we face the multifaceted challenges of the 21st century, transitioning to renewable energy emerges as not just an option but an essential step toward a sustainable future...."
        }
    ],
    "number_of_agents": 4,
    "service_tier": "standard",
    "execution_time": 72.23084282875061,
    "usage": {
        "input_tokens": 28,
        "output_tokens": 3012,
        "total_tokens": 3040,
        "billing_info": {
            "cost_breakdown": {
                "agent_cost": 0.04,
                "input_token_cost": 0.000084,
                "output_token_cost": 0.04518,
                "token_counts": {
                    "total_input_tokens": 28,
                    "total_output_tokens": 3012,
                    "total_tokens": 3040
                },
                "num_agents": 4,
                "service_tier": "standard",
                "night_time_discount_applied": true
            },
            "total_cost": 0.085264,
            "discount_active": true,
            "discount_type": "night_time",
            "discount_percentage": 75
        }
    }
}
```

## Best Practices

- Design agents with clear, sequential dependencies
- Ensure each agent builds meaningfully on the previous work
- Use for linear workflows where order matters
- Validate outputs at each step before proceeding

## Related Swarm Types

- [ConcurrentWorkflow](concurrent_workflow.md) - For parallel execution
- [AgentRearrange](agent_rearrange.md) - For dynamic sequencing
- [HierarchicalSwarm](hierarchical_swarm.md) - For structured workflows