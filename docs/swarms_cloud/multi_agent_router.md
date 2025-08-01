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
  "status": "success",
  "swarm_name": "customer-support-router",
  "swarm_type": "MultiAgentRouter",
  "task": "Handle multiple customer inquiries: 1) Billing question about overcharge, 2) Technical issue with mobile app login, 3) Product recommendation for enterprise client, 4) Return policy question",
  "output": {
    "inquiry_1_billing": {
      "routed_to": "Billing Specialist",
      "response": "I understand your concern about the overcharge. Let me review your account and identify the issue. I can see the duplicate charge and will process a refund within 3-5 business days...",
      "resolution_status": "Resolved - Refund processed"
    },
    "inquiry_2_technical": {
      "routed_to": "Technical Support",
      "response": "Let's troubleshoot the mobile app login issue. Please try these steps: 1) Clear app cache, 2) Update to latest version, 3) Reset password if needed...",
      "resolution_status": "In Progress - Troubleshooting steps provided"
    },
    "inquiry_3_sales": {
      "routed_to": "Sales Consultant", 
      "response": "For enterprise clients, I recommend our Professional tier with advanced analytics, dedicated support, and custom integrations. This includes...",
      "resolution_status": "Proposal sent - Follow-up scheduled"
    },
    "inquiry_4_policy": {
      "routed_to": "Policy Advisor",
      "response": "Our return policy allows returns within 30 days of purchase for full refund. Items must be in original condition. Here's the complete process...",
      "resolution_status": "Information provided - Customer satisfied"
    }
  },
  "metadata": {
    "routing_decisions": [
      {
        "inquiry": "Billing question about overcharge",
        "routed_to": "Billing Specialist",
        "confidence": 0.95,
        "reasoning": "Billing-related inquiry requires specialized financial expertise"
      },
      {
        "inquiry": "Technical issue with mobile app login",
        "routed_to": "Technical Support",
        "confidence": 0.98,
        "reasoning": "Technical troubleshooting requires technical specialist"
      },
      {
        "inquiry": "Product recommendation for enterprise client",
        "routed_to": "Sales Consultant",
        "confidence": 0.92,
        "reasoning": "Enterprise sales requires specialized sales expertise"
      },
      {
        "inquiry": "Return policy question",
        "routed_to": "Policy Advisor",
        "confidence": 0.97,
        "reasoning": "Policy questions require policy specialist knowledge"
      }
    ],
    "routing_efficiency": "100% - All inquiries routed to optimal agents",
    "execution_time_seconds": 16.4,
    "billing_info": {
      "total_cost": 0.042
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