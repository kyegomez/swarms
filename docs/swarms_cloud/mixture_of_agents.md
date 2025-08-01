# MixtureOfAgents

*Builds diverse teams of specialized agents for complex problem solving*

**Swarm Type**: `MixtureOfAgents`

## Overview

The MixtureOfAgents swarm type combines multiple agent types with different specializations to tackle diverse aspects of complex problems. Each agent contributes unique skills and perspectives, making this architecture ideal for tasks requiring multiple types of expertise working in harmony.

Key features:
- **Diverse Expertise**: Combines agents with different specializations
- **Collaborative Problem Solving**: Agents work together leveraging their unique strengths
- **Comprehensive Coverage**: Ensures all aspects of complex tasks are addressed
- **Balanced Perspectives**: Multiple viewpoints for robust decision-making

## Use Cases

- Complex research projects requiring multiple disciplines
- Business analysis needing various functional perspectives
- Content creation requiring different expertise areas
- Strategic planning with multiple considerations

## API Usage

### Basic MixtureOfAgents Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Business Strategy Mixture",
        "description": "Diverse team analyzing business strategy from multiple perspectives",
        "swarm_type": "MixtureOfAgents",
        "task": "Develop a comprehensive market entry strategy for a new AI product in the healthcare sector",
        "agents": [
          {
            "agent_name": "Market Research Analyst",
            "description": "Analyzes market trends and opportunities",
            "system_prompt": "You are a market research expert specializing in healthcare technology. Analyze market size, trends, and competitive landscape.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Financial Analyst",
            "description": "Evaluates financial viability and projections",
            "system_prompt": "You are a financial analyst expert. Assess financial implications, ROI, and cost structures for business strategies.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.2
          },
          {
            "agent_name": "Regulatory Expert",
            "description": "Analyzes compliance and regulatory requirements",
            "system_prompt": "You are a healthcare regulatory expert. Analyze compliance requirements, regulatory pathways, and potential barriers.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.1
          },
          {
            "agent_name": "Technology Strategist",
            "description": "Evaluates technical feasibility and strategy",
            "system_prompt": "You are a technology strategy expert. Assess technical requirements, implementation challenges, and scalability.",
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
        "name": "Business Strategy Mixture",
        "description": "Diverse team analyzing business strategy from multiple perspectives",
        "swarm_type": "MixtureOfAgents",
        "task": "Develop a comprehensive market entry strategy for a new AI product in the healthcare sector",
        "agents": [
            {
                "agent_name": "Market Research Analyst",
                "description": "Analyzes market trends and opportunities",
                "system_prompt": "You are a market research expert specializing in healthcare technology. Analyze market size, trends, and competitive landscape.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Financial Analyst",
                "description": "Evaluates financial viability and projections",
                "system_prompt": "You are a financial analyst expert. Assess financial implications, ROI, and cost structures for business strategies.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.2
            },
            {
                "agent_name": "Regulatory Expert",
                "description": "Analyzes compliance and regulatory requirements",
                "system_prompt": "You are a healthcare regulatory expert. Analyze compliance requirements, regulatory pathways, and potential barriers.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.1
            },
            {
                "agent_name": "Technology Strategist",
                "description": "Evaluates technical feasibility and strategy",
                "system_prompt": "You are a technology strategy expert. Assess technical requirements, implementation challenges, and scalability.",
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
        print("MixtureOfAgents swarm completed successfully!")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Output: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
    "job_id": "swarms-kBZaJg1uGTkRbLCAsGztL2jrp5Mj",
    "status": "success",
    "swarm_name": "Business Strategy Mixture",
    "description": "Diverse team analyzing business strategy from multiple perspectives",
    "swarm_type": "MixtureOfAgents",
    "output": [
        {
            "role": "System",
            "content": "Team Name: Business Strategy Mixture\nTeam Description: Diverse team analyzing business strategy from multiple perspectives\nThese are the agents in your team. Each agent has a specific role and expertise to contribute to the team's objectives.\nTotal Agents: 4\n\nBelow is a summary of your team members and their primary responsibilities:\n| Agent Name | Description |\n|------------|-------------|\n| Market Research Analyst | Analyzes market trends and opportunities |\n| Financial Analyst | Evaluates financial viability and projections |\n| Regulatory Expert | Analyzes compliance and regulatory requirements |\n| Technology Strategist | Evaluates technical feasibility and strategy |\n\nEach agent is designed to handle tasks within their area of expertise. Collaborate effectively by assigning tasks according to these roles."
        },
        {
            "role": "Market Research Analyst",
            "content": "To develop a comprehensive market entry strategy for a new AI product in the healthcare sector, we will leverage the expertise of each team member to cover all critical aspects of the strategy. Here's how each agent will contribute......."
        },
        {
            "role": "Technology Strategist",
            "content": "To develop a comprehensive market entry strategy for a new AI product in the healthcare sector, we'll need to collaborate effectively with the team, leveraging each member's expertise. Here's how each agent can contribute to the strategy, along with a focus on the technical requirements, implementation challenges, and scalability from the technology strategist's perspective....."
        },
        {
            "role": "Financial Analyst",
            "content": "Developing a comprehensive market entry strategy for a new AI product in the healthcare sector involves a multidisciplinary approach. Each agent in the Business Strategy Mixture team will play a crucial role in ensuring a successful market entry. Here's how the team can collaborate........"
        },
        {
            "role": "Regulatory Expert",
            "content": "To develop a comprehensive market entry strategy for a new AI product in the healthcare sector, we need to leverage the expertise of each agent in the Business Strategy Mixture team. Below is an outline of how each team member can contribute to this strategy......"
        },
        {
            "role": "Aggregator Agent",
            "content": "As the Aggregator Agent, I've observed and analyzed the responses from the Business Strategy Mixture team regarding the development of a comprehensive market entry strategy for a new AI product in the healthcare sector. Here's a summary of the key points ......"
        }
    ],
    "number_of_agents": 4,
    "service_tier": "standard",
    "execution_time": 30.230480670928955,
    "usage": {
        "input_tokens": 30,
        "output_tokens": 3401,
        "total_tokens": 3431,
        "billing_info": {
            "cost_breakdown": {
                "agent_cost": 0.04,
                "input_token_cost": 0.00009,
                "output_token_cost": 0.051015,
                "token_counts": {
                    "total_input_tokens": 30,
                    "total_output_tokens": 3401,
                    "total_tokens": 3431
                },
                "num_agents": 4,
                "service_tier": "standard",
                "night_time_discount_applied": true
            },
            "total_cost": 0.091105,
            "discount_active": true,
            "discount_type": "night_time",
            "discount_percentage": 75
        }
    }
}
```

## Best Practices

- Select agents with complementary and diverse expertise
- Ensure each agent has a clear, specialized role
- Use for complex problems requiring multiple perspectives
- Design tasks that benefit from collaborative analysis

## Related Swarm Types

- [ConcurrentWorkflow](concurrent_workflow.md) - For parallel task execution
- [GroupChat](group_chat.md) - For collaborative discussion
- [AutoSwarmBuilder](auto_swarm_builder.md) - For automatic team assembly