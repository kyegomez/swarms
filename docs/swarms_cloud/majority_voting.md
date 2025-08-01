# MajorityVoting

*Implements robust decision-making through consensus and voting*

**Swarm Type**: `MajorityVoting`

## Overview

The MajorityVoting swarm type implements robust decision-making through consensus mechanisms, ideal for tasks requiring collective intelligence or verification. Multiple agents independently analyze the same problem and vote on the best solution, ensuring high-quality, well-validated outcomes through democratic consensus.

Key features:
- **Consensus-Based Decisions**: Multiple agents vote on the best solution
- **Quality Assurance**: Reduces individual agent bias through collective input
- **Democratic Process**: Fair and transparent decision-making mechanism
- **Robust Validation**: Multiple perspectives ensure thorough analysis

## Use Cases

- Critical decision-making requiring validation
- Quality assurance and verification tasks
- Complex problem solving with multiple viable solutions
- Risk assessment and evaluation scenarios

## API Usage

### Basic MajorityVoting Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Investment Decision Voting",
        "description": "Multiple financial experts vote on investment recommendations",
        "swarm_type": "MajorityVoting",
        "task": "Evaluate whether to invest $1M in a renewable energy startup. Consider market potential, financial projections, team strength, and competitive landscape.",
        "agents": [
          {
            "agent_name": "Growth Investor",
            "description": "Focuses on growth potential and market opportunity",
            "system_prompt": "You are a growth-focused venture capitalist. Evaluate investments based on market size, scalability, and growth potential. Provide a recommendation with confidence score.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Financial Analyst",
            "description": "Analyzes financial metrics and projections",
            "system_prompt": "You are a financial analyst specializing in startups. Evaluate financial projections, revenue models, and unit economics. Provide a recommendation with confidence score.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.2
          },
          {
            "agent_name": "Technical Due Diligence",
            "description": "Evaluates technology and product viability",
            "system_prompt": "You are a technical due diligence expert. Assess technology viability, intellectual property, product-market fit, and technical risks. Provide a recommendation with confidence score.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Market Analyst",
            "description": "Analyzes market conditions and competition",
            "system_prompt": "You are a market research analyst. Evaluate market dynamics, competitive landscape, regulatory environment, and market timing. Provide a recommendation with confidence score.",
            "model_name": "gpt-4o",
            "max_loops": 1,
            "temperature": 0.3
          },
          {
            "agent_name": "Risk Assessor",
            "description": "Identifies and evaluates investment risks",
            "system_prompt": "You are a risk assessment specialist. Identify potential risks, evaluate mitigation strategies, and assess overall risk profile. Provide a recommendation with confidence score.",
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
        "name": "Investment Decision Voting",
        "description": "Multiple financial experts vote on investment recommendations",
        "swarm_type": "MajorityVoting",
        "task": "Evaluate whether to invest $1M in a renewable energy startup. Consider market potential, financial projections, team strength, and competitive landscape.",
        "agents": [
            {
                "agent_name": "Growth Investor",
                "description": "Focuses on growth potential and market opportunity",
                "system_prompt": "You are a growth-focused venture capitalist. Evaluate investments based on market size, scalability, and growth potential. Provide a recommendation with confidence score.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Financial Analyst",
                "description": "Analyzes financial metrics and projections",
                "system_prompt": "You are a financial analyst specializing in startups. Evaluate financial projections, revenue models, and unit economics. Provide a recommendation with confidence score.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.2
            },
            {
                "agent_name": "Technical Due Diligence",
                "description": "Evaluates technology and product viability",
                "system_prompt": "You are a technical due diligence expert. Assess technology viability, intellectual property, product-market fit, and technical risks. Provide a recommendation with confidence score.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Market Analyst",
                "description": "Analyzes market conditions and competition",
                "system_prompt": "You are a market research analyst. Evaluate market dynamics, competitive landscape, regulatory environment, and market timing. Provide a recommendation with confidence score.",
                "model_name": "gpt-4o",
                "max_loops": 1,
                "temperature": 0.3
            },
            {
                "agent_name": "Risk Assessor",
                "description": "Identifies and evaluates investment risks",
                "system_prompt": "You are a risk assessment specialist. Identify potential risks, evaluate mitigation strategies, and assess overall risk profile. Provide a recommendation with confidence score.",
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
        print("MajorityVoting completed successfully!")
        print(f"Final decision: {result['output']['consensus_decision']}")
        print(f"Vote breakdown: {result['metadata']['vote_breakdown']}")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
  "status": "success",
  "swarm_name": "investment-decision-voting",
  "swarm_type": "MajorityVoting",
  "task": "Evaluate whether to invest $1M in a renewable energy startup. Consider market potential, financial projections, team strength, and competitive landscape.",
  "output": {
    "individual_recommendations": [
      {
        "agent": "Growth Investor",
        "recommendation": "INVEST",
        "confidence": 0.8,
        "reasoning": "Strong market growth potential in renewable energy sector, scalable technology platform"
      },
      {
        "agent": "Financial Analyst", 
        "recommendation": "INVEST",
        "confidence": 0.7,
        "reasoning": "Solid financial projections, reasonable burn rate, clear path to profitability"
      },
      {
        "agent": "Technical Due Diligence",
        "recommendation": "INVEST",
        "confidence": 0.75,
        "reasoning": "Innovative technology with strong IP portfolio, experienced technical team"
      },
      {
        "agent": "Market Analyst",
        "recommendation": "WAIT",
        "confidence": 0.6,
        "reasoning": "Highly competitive market, regulatory uncertainties may impact timeline"
      },
      {
        "agent": "Risk Assessor",
        "recommendation": "INVEST",
        "confidence": 0.65,
        "reasoning": "Manageable risks with strong mitigation strategies, experienced leadership team"
      }
    ],
    "consensus_decision": "INVEST",
    "consensus_confidence": 0.72,
    "consensus_reasoning": "4 out of 5 experts recommend investment with strong market potential and solid fundamentals, despite some market uncertainties"
  },
  "metadata": {
    "vote_breakdown": {
      "INVEST": 4,
      "WAIT": 1,
      "REJECT": 0
    },
    "vote_percentage": {
      "INVEST": "80%",
      "WAIT": "20%",
      "REJECT": "0%"
    },
    "average_confidence": 0.70,
    "consensus_threshold": "Simple majority (50%+)",
    "execution_time_seconds": 25.8,
    "billing_info": {
      "total_cost": 0.063
    }
  }
}
```

## Best Practices

- Use odd numbers of agents to avoid tie votes
- Design agents with different perspectives for robust evaluation
- Include confidence scores in agent prompts for weighted decisions
- Ideal for high-stakes decisions requiring validation

## Related Swarm Types

- [GroupChat](group_chat.md) - For discussion-based consensus
- [MixtureOfAgents](mixture_of_agents.md) - For diverse expertise collaboration
- [HierarchicalSwarm](hierarchical_swarm.md) - For structured decision-making