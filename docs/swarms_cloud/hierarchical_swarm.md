# HierarchicalSwarm

*Implements structured, multi-level task management with clear authority*

**Swarm Type**: `HierarchicalSwarm`

## Overview

The HierarchicalSwarm implements a structured, multi-level approach to task management with clear lines of authority and delegation. This architecture organizes agents in a hierarchical structure where manager agents coordinate and oversee worker agents, enabling efficient task distribution and quality control.

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

### Basic HierarchicalSwarm Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Software Development Hierarchy",
        "description": "Hierarchical software development team with project manager oversight",
        "swarm_type": "HierarchicalSwarm",
        "task": "Design and plan a new mobile app for expense tracking targeting freelancers",
        "agents": [
          {
            "agent_name": "Project Manager",
            "description": "Oversees project planning and coordinates team efforts",
            "system_prompt": "You are a senior project manager. Coordinate the team, break down tasks, ensure quality, and synthesize outputs. Delegate specific tasks to team members.",
            "model_name": "gpt-4o",
            "role": "manager",
            "max_loops": 2,
            "temperature": 0.4
          },
          {
            "agent_name": "UX Designer",
            "description": "Designs user experience and interface",
            "system_prompt": "You are a UX designer specializing in mobile apps. Focus on user flows, wireframes, and interface design. Report findings to the project manager.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "temperature": 0.6
          },
          {
            "agent_name": "Technical Architect",
            "description": "Designs technical architecture and system requirements",
            "system_prompt": "You are a technical architect. Focus on system design, technology stack, database design, and technical requirements. Provide technical guidance.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Business Analyst",
            "description": "Analyzes business requirements and market fit",
            "system_prompt": "You are a business analyst. Focus on requirements gathering, market analysis, feature prioritization, and business logic.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "temperature": 0.4
          },
          {
            "agent_name": "QA Specialist",
            "description": "Ensures quality and validates deliverables",
            "system_prompt": "You are a QA specialist. Review all outputs for quality, completeness, and consistency. Identify gaps and suggest improvements.",
            "model_name": "gpt-4o",
            "role": "worker",
            "max_loops": 1,
            "temperature": 0.2
          }
        ],
        "max_loops": 2
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
        "name": "Software Development Hierarchy",
        "description": "Hierarchical software development team with project manager oversight",
        "swarm_type": "HierarchicalSwarm",
        "task": "Design and plan a new mobile app for expense tracking targeting freelancers",
        "agents": [
            {
                "agent_name": "Project Manager",
                "description": "Oversees project planning and coordinates team efforts",
                "system_prompt": "You are a senior project manager. Coordinate the team, break down tasks, ensure quality, and synthesize outputs. Delegate specific tasks to team members.",
                "model_name": "gpt-4o",
                "role": "manager",
                "max_loops": 2,
                "temperature": 0.4
            },
            {
                "agent_name": "UX Designer",
                "description": "Designs user experience and interface",
                "system_prompt": "You are a UX designer specializing in mobile apps. Focus on user flows, wireframes, and interface design. Report findings to the project manager.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "temperature": 0.6
            },
            {
                "agent_name": "Technical Architect",
                "description": "Designs technical architecture and system requirements",
                "system_prompt": "You are a technical architect. Focus on system design, technology stack, database design, and technical requirements. Provide technical guidance.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Business Analyst",
                "description": "Analyzes business requirements and market fit",
                "system_prompt": "You are a business analyst. Focus on requirements gathering, market analysis, feature prioritization, and business logic.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "temperature": 0.4
            },
            {
                "agent_name": "QA Specialist",
                "description": "Ensures quality and validates deliverables",
                "system_prompt": "You are a QA specialist. Review all outputs for quality, completeness, and consistency. Identify gaps and suggest improvements.",
                "model_name": "gpt-4o",
                "role": "worker",
                "max_loops": 1,
                "temperature": 0.2
            }
        ],
        "max_loops": 2
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print("HierarchicalSwarm completed successfully!")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Project plan: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
  "status": "success",
  "swarm_name": "software-development-hierarchy",
  "swarm_type": "HierarchicalSwarm",
  "task": "Design and plan a new mobile app for expense tracking targeting freelancers",
  "output": {
    "project_overview": {
      "manager_synthesis": "Comprehensive project plan for freelancer expense tracking app...",
      "timeline": "16-week development cycle",
      "key_deliverables": ["UX Design", "Technical Architecture", "Business Requirements", "QA Framework"]
    },
    "ux_design": {
      "user_flows": "Streamlined expense entry and categorization flows...",
      "wireframes": "Mobile-first design with dashboard and reporting views...",
      "usability_considerations": "One-tap expense entry, photo receipt capture..."
    },
    "technical_architecture": {
      "tech_stack": "React Native, Node.js, PostgreSQL, AWS",
      "system_design": "Microservices architecture with offline capability...",
      "security_requirements": "End-to-end encryption, secure authentication..."
    },
    "business_requirements": {
      "target_market": "Freelancers and independent contractors", 
      "core_features": ["Expense tracking", "Receipt scanning", "Tax reporting"],
      "monetization": "Freemium model with premium reporting features"
    },
    "qa_framework": {
      "testing_strategy": "Automated testing for core functions...",
      "quality_metrics": "Performance, usability, and security benchmarks...",
      "validation_checkpoints": "Weekly reviews and milestone validations"
    }
  },
  "metadata": {
    "hierarchy_structure": {
      "managers": ["Project Manager"],
      "workers": ["UX Designer", "Technical Architect", "Business Analyst", "QA Specialist"]
    },
    "coordination_rounds": 2,
    "task_delegation": "Manager coordinated 4 specialized work streams",
    "execution_time_seconds": 52.4,
    "billing_info": {
      "total_cost": 0.128
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