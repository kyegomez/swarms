# HiearchicalSwarm

*Implements structured, multi-level task management with clear authority*

**Swarm Type**: `HiearchicalSwarm`

## Overview

The HiearchicalSwarm implements a structured, multi-level approach to task management with clear lines of authority and delegation. This architecture organizes agents in a hierarchical structure where manager agents coordinate and oversee worker agents, enabling efficient task distribution and quality control.

Key features:
- **Structured Hierarchy**: Clear organizational structure with managers and workers
- **Delegated Authority**: Manager agents distribute tasks to specialized workers
- **Quality Oversight**: Multi-level review and validation processes
- **Scalable Organization**: Efficient coordination of large agent teams

## Use Cases

- Complex projects requiring management oversight
- Large-scale content production workflows
- Multi-stage validation and review processes
- Enterprise-level task coordination

## API Usage

### Basic HiearchicalSwarm Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Market Research ",
        "description": "Parallel market research across different sectors",
        "swarm_type": "HiearchicalSwarm",
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
        "name": "Market Research ",
        "description": "Parallel market research across different sectors",
        "swarm_type": "HiearchicalSwarm",
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
        print("HiearchicalSwarm completed successfully!")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Project plan: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
    "job_id": "swarms-JIrcIAfs2d75xrXGaAL94uWyYJ8V",
    "status": "success",
    "swarm_name": "Market Research Auto",
    "description": "Parallel market research across different sectors",
    "swarm_type": "HiearchicalSwarm",
    "output": [
        {
            "role": "System",
            "content": "These are the agents in your team. Each agent has a specific role and expertise to contribute to the team's objectives.\nTotal Agents: 4\n\nBelow is a summary of your team members and their primary responsibilities:\n| Agent Name | Description |\n|------------|-------------|\n| AI Market Analyst | Analyzes AI market trends and opportunities |\n| Healthcare Market Analyst | Analyzes healthcare market trends |\n| Fintech Market Analyst | Analyzes fintech market opportunities |\n| E-commerce Market Analyst | Analyzes e-commerce market trends |\n\nEach agent is designed to handle tasks within their area of expertise. Collaborate effectively by assigning tasks according to these roles."
        },
        {
            "role": "Director",
            "content": [
                {
                    "role": "Director",
                    "content": [
                        {
                            "function": {
                                "arguments": "{\"plan\":\"Conduct a comprehensive analysis of market opportunities in the AI, healthcare, fintech, and e-commerce sectors. Each market analyst will focus on their respective sector, gathering data on current trends, growth opportunities, and potential challenges. The findings will be compiled into a report for strategic decision-making.\",\"orders\":[{\"agent_name\":\"AI Market Analyst\",\"task\":\"Research current trends in the AI market, identify growth opportunities, and analyze potential challenges.\"},{\"agent_name\":\"Healthcare Market Analyst\",\"task\":\"Analyze the healthcare market for emerging trends, growth opportunities, and possible challenges.\"},{\"agent_name\":\"Fintech Market Analyst\",\"task\":\"Investigate the fintech sector for current trends, identify opportunities for growth, and assess challenges.\"},{\"agent_name\":\"E-commerce Market Analyst\",\"task\":\"Examine e-commerce market trends, identify growth opportunities, and analyze potential challenges.\"}]}",
                                "name": "ModelMetaclass"
                            },
                            "id": "call_GxiyzIRb2oGQXokbbkeaeVry",
                            "type": "function"
                        }
                    ]
                }
            ]
        },
        {
            "role": "AI Market Analyst",
            "content": "### AI Market Analysis: Trends, Opportunities, and Challenges\n\n#### Current Trends in the AI Market:\n\n1. **Increased Adoption Across Industries**..."
        },
        {
            "role": "Healthcare Market Analyst",
            "content": "### Healthcare Market Analysis: Trends, Opportunities, and Challenges\n\n#### Current Trends in the Healthcare Market:\n\n1. **Telehealth Expansion**..."
        },
        {
            "role": "Fintech Market Analyst",
            "content": "### Fintech Market Analysis: Trends, Opportunities, and Challenges\n\n#### Current Trends in the Fintech Market:\n\n1. **Digital Payments Proliferation**...."
        },
        {
            "role": "E-commerce Market Analyst",
            "content": "### E-commerce Market Analysis: Trends, Opportunities, and Challenges\n\n#### Current Trends in the E-commerce Market:\n\n1. **Omnichannel Retailing**...."
        },
        {
            "role": "Director",
            "content": "### Feedback for Worker Agents\n\n#### AI Market Analyst\n\n**Strengths:**\n- Comprehensive coverage of current trends, growth opportunities, and challenges in the AI market.\n- Clear categorization of insights, making it easy to follow and understand.\n\n**Weaknesses....."
        },
        {
            "role": "System",
            "content": "--- Loop 1/1 completed ---"
        }
    ],
    "number_of_agents": 4,
    "service_tier": "standard",
    "execution_time": 94.07934331893921,
    "usage": {
        "input_tokens": 35,
        "output_tokens": 3827,
        "total_tokens": 3862,
        "billing_info": {
            "cost_breakdown": {
                "agent_cost": 0.04,
                "input_token_cost": 0.000105,
                "output_token_cost": 0.057405,
                "token_counts": {
                    "total_input_tokens": 35,
                    "total_output_tokens": 3827,
                    "total_tokens": 3862
                },
                "num_agents": 4,
                "service_tier": "standard",
                "night_time_discount_applied": false
            },
            "total_cost": 0.09751,
            "discount_active": false,
            "discount_type": "none",
            "discount_percentage": 0
        }
    }
}
```

## Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `role` | string | Agent role: "manager" or "worker" | "worker" |
| `agents` | Array<AgentSpec> | Mix of manager and worker agents | Required |
| `max_loops` | integer | Coordination rounds for managers | 1 |

## Best Practices

- Clearly define manager and worker roles using the `role` parameter
- Give managers higher `max_loops` for coordination activities
- Design worker agents with specialized, focused responsibilities
- Use for complex projects requiring oversight and coordination

## Related Swarm Types

- [SequentialWorkflow](sequential_workflow.md) - For linear task progression
- [MultiAgentRouter](multi_agent_router.md) - For intelligent task routing
- [AutoSwarmBuilder](auto_swarm_builder.md) - For automatic hierarchy creation