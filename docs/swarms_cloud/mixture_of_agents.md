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
  "status": "success",
  "swarm_name": "business-strategy-mixture",
  "swarm_type": "MixtureOfAgents",
  "task": "Develop a comprehensive market entry strategy for a new AI product in the healthcare sector",
  "output": {
    "market_analysis": "Detailed market research findings...",
    "financial_assessment": "Financial projections and ROI analysis...",
    "regulatory_compliance": "Regulatory requirements and pathways...",
    "technology_strategy": "Technical implementation roadmap...",
    "integrated_strategy": "Comprehensive market entry strategy combining all perspectives..."
  },
  "metadata": {
    "agent_contributions": {
      "Market Research Analyst": "Market size: $2.3B, Growth rate: 15% CAGR",
      "Financial Analyst": "Break-even: 18 months, ROI: 35%",
      "Regulatory Expert": "FDA pathway: 510(k), Timeline: 8-12 months",
      "Technology Strategist": "MVP timeline: 6 months, Scalability: High"
    },
    "execution_time_seconds": 22.1,
    "billing_info": {
      "total_cost": 0.067
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