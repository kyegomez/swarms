# MultiAgentRouter

*Intelligent task dispatcher distributing work based on agent capabilities*

**Swarm Type**: `MultiAgentRouter`

## Overview

The MultiAgentRouter acts as an intelligent task dispatcher, distributing work across agents based on their capabilities and current workload. This architecture analyzes incoming tasks and automatically routes them to the most suitable agents, optimizing both efficiency and quality of outcomes.

Key features:
- **Intelligent Routing**: Automatically assigns tasks to best-suited agents
- **Capability Matching**: Matches task requirements with agent specializations
- **Load Balancing**: Distributes workload efficiently across available agents
- **Dynamic Assignment**: Adapts routing based on agent performance and availability

## Use Cases

- Customer service request routing
- Content categorization and processing
- Technical support ticket assignment
- Multi-domain question answering

## API Usage

### Basic MultiAgentRouter Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Customer Support Router",
        "description": "Route customer inquiries to specialized support agents",
        "swarm_type": "MultiAgentRouter",
        "task": "Handle multiple customer inquiries: 1) Billing question about overcharge, 2) Technical issue with mobile app login, 3) Product recommendation for enterprise client, 4) Return policy question",
        "agents": [
          {
            "agent_name": "Billing Specialist",
            "description": "Handles billing, payments, and account issues",
            "system_prompt": "You are a billing specialist. Handle all billing inquiries, payment issues, refunds, and account-related questions with empathy and accuracy.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Technical Support",
            "description": "Resolves technical issues and troubleshooting",
            "system_prompt": "You are a technical support specialist. Diagnose and resolve technical issues, provide step-by-step troubleshooting, and escalate complex problems.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.2
          },
          {
            "agent_name": "Sales Consultant",
            "description": "Provides product recommendations and sales support",
            "system_prompt": "You are a sales consultant. Provide product recommendations, explain features and benefits, and help customers find the right solutions.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.4
          },
          {
            "agent_name": "Policy Advisor",
            "description": "Explains company policies and procedures",
            "system_prompt": "You are a policy advisor. Explain company policies, terms of service, return procedures, and compliance requirements clearly.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.1
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
        "name": "Customer Support Router",
        "description": "Route customer inquiries to specialized support agents",
        "swarm_type": "MultiAgentRouter",
        "task": "Handle multiple customer inquiries: 1) Billing question about overcharge, 2) Technical issue with mobile app login, 3) Product recommendation for enterprise client, 4) Return policy question",
        "agents": [
            {
                "agent_name": "Billing Specialist",
                "description": "Handles billing, payments, and account issues",
                "system_prompt": "You are a billing specialist. Handle all billing inquiries, payment issues, refunds, and account-related questions with empathy and accuracy.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Technical Support",
                "description": "Resolves technical issues and troubleshooting",
                "system_prompt": "You are a technical support specialist. Diagnose and resolve technical issues, provide step-by-step troubleshooting, and escalate complex problems.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.2
            },
            {
                "agent_name": "Sales Consultant",
                "description": "Provides product recommendations and sales support",
                "system_prompt": "You are a sales consultant. Provide product recommendations, explain features and benefits, and help customers find the right solutions.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.4
            },
            {
                "agent_name": "Policy Advisor",
                "description": "Explains company policies and procedures",
                "system_prompt": "You are a policy advisor. Explain company policies, terms of service, return procedures, and compliance requirements clearly.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.1
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
        print("MultiAgentRouter completed successfully!")
        print(f"Routing decisions: {result['metadata']['routing_decisions']}")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Customer responses: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
    "job_id": "swarms-OvOZHubprE3thzLmRdNBZAxA6om4",
    "status": "success",
    "swarm_name": "Customer Support Router",
    "description": "Route customer inquiries to specialized support agents",
    "swarm_type": "MultiAgentRouter",
    "output": [
        {
            "role": "user",
            "content": "Handle multiple customer inquiries: 1) Billing question about overcharge, 2) Technical issue with mobile app login, 3) Product recommendation for enterprise client, 4) Return policy question"
        },
        {
            "role": "Agent Router",
            "content": "selected_agent='Billing Specialist' reasoning='The task involves multiple inquiries, but the first one is about a billing question regarding an overcharge. Billing issues often require immediate attention to ensure customer satisfaction and prevent further complications. Therefore, the Billing Specialist is the most appropriate agent to handle this task. They can address the billing question directly and potentially coordinate with other agents for the remaining inquiries.' modified_task='Billing question about overcharge'"
        },
        {
            "role": "Billing Specialist",
            "content": "Of course, I'd be happy to help you with your billing question regarding an overcharge. Could you please provide me with more details about the charge in question, such as the date it occurred and the amount? This information will help me look into your account and resolve the issue as quickly as possible."
        }
    ],
    "number_of_agents": 4,
    "service_tier": "standard",
    "execution_time": 7.800086975097656,
    "usage": {
        "input_tokens": 28,
        "output_tokens": 221,
        "total_tokens": 249,
        "billing_info": {
            "cost_breakdown": {
                "agent_cost": 0.04,
                "input_token_cost": 0.000084,
                "output_token_cost": 0.003315,
                "token_counts": {
                    "total_input_tokens": 28,
                    "total_output_tokens": 221,
                    "total_tokens": 249
                },
                "num_agents": 4,
                "service_tier": "standard",
                "night_time_discount_applied": true
            },
            "total_cost": 0.043399,
            "discount_active": true,
            "discount_type": "night_time",
            "discount_percentage": 75
        }
    }
}
```

## Best Practices

- Define agents with clear, distinct specializations
- Use descriptive agent names and descriptions for better routing
- Ideal for handling diverse task types that require different expertise
- Monitor routing decisions to optimize agent configurations

## Related Swarm Types

- [HierarchicalSwarm](hierarchical_swarm.md) - For structured task management
- [ConcurrentWorkflow](concurrent_workflow.md) - For parallel task processing
- [AutoSwarmBuilder](auto_swarm_builder.md) - For automatic routing setup