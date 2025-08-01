# GroupChat

*Enables dynamic collaboration through chat-based interaction*

**Swarm Type**: `GroupChat`

## Overview

The GroupChat swarm type enables dynamic collaboration between agents through a chat-based interface, facilitating real-time information sharing and decision-making. Agents participate in a conversational workflow where they can build upon each other's contributions, debate ideas, and reach consensus through natural dialogue.

Key features:
- **Interactive Dialogue**: Agents communicate through natural conversation
- **Dynamic Collaboration**: Real-time information sharing and building upon ideas  
- **Consensus Building**: Agents can debate and reach decisions collectively
- **Flexible Participation**: Agents can contribute when relevant to the discussion

## Use Cases

- Brainstorming and ideation sessions
- Multi-perspective problem analysis
- Collaborative decision-making processes
- Creative content development

## API Usage

### Basic GroupChat Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Product Strategy Discussion",
        "description": "Collaborative chat to develop product strategy",
        "swarm_type": "GroupChat",
        "task": "Discuss and develop a go-to-market strategy for a new AI-powered productivity tool targeting small businesses",
        "agents": [
          {
            "agent_name": "Product Manager",
            "description": "Leads product strategy and development",
            "system_prompt": "You are a senior product manager. Focus on product positioning, features, user needs, and market fit. Ask probing questions and build on others ideas.",
            "model_name": "gpt-4o",
            "max_loops": 3,
            "temperature": 0.6
          },
          {
            "agent_name": "Marketing Strategist", 
            "description": "Develops marketing and positioning strategy",
            "system_prompt": "You are a marketing strategist. Focus on target audience, messaging, channels, and competitive positioning. Contribute marketing insights to the discussion.",
            "model_name": "gpt-4o",
            "max_loops": 3,
            "temperature": 0.7
          },
          {
            "agent_name": "Sales Director",
            "description": "Provides sales and customer perspective",
            "system_prompt": "You are a sales director with small business experience. Focus on pricing, sales process, customer objections, and market adoption. Share practical sales insights.",
            "model_name": "gpt-4o",
            "max_loops": 3,
            "temperature": 0.5
          },
          {
            "agent_name": "UX Researcher",
            "description": "Represents user experience and research insights",
            "system_prompt": "You are a UX researcher specializing in small business tools. Focus on user behavior, usability, adoption barriers, and design considerations.",
            "model_name": "gpt-4o",
            "max_loops": 3,
            "temperature": 0.4
          }
        ],
        "max_loops": 3
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
        "name": "Product Strategy Discussion",
        "description": "Collaborative chat to develop product strategy",
        "swarm_type": "GroupChat",
        "task": "Discuss and develop a go-to-market strategy for a new AI-powered productivity tool targeting small businesses",
        "agents": [
            {
                "agent_name": "Product Manager",
                "description": "Leads product strategy and development",
                "system_prompt": "You are a senior product manager. Focus on product positioning, features, user needs, and market fit. Ask probing questions and build on others ideas.",
                "model_name": "gpt-4o",
                "max_loops": 3,
                "temperature": 0.6
            },
            {
                "agent_name": "Marketing Strategist", 
                "description": "Develops marketing and positioning strategy",
                "system_prompt": "You are a marketing strategist. Focus on target audience, messaging, channels, and competitive positioning. Contribute marketing insights to the discussion.",
                "model_name": "gpt-4o",
                "max_loops": 3,
                "temperature": 0.7
            },
            {
                "agent_name": "Sales Director",
                "description": "Provides sales and customer perspective",
                "system_prompt": "You are a sales director with small business experience. Focus on pricing, sales process, customer objections, and market adoption. Share practical sales insights.",
                "model_name": "gpt-4o",
                "max_loops": 3,
                "temperature": 0.5
            },
            {
                "agent_name": "UX Researcher",
                "description": "Represents user experience and research insights",
                "system_prompt": "You are a UX researcher specializing in small business tools. Focus on user behavior, usability, adoption barriers, and design considerations.",
                "model_name": "gpt-4o",
                "max_loops": 3,
                "temperature": 0.4
            }
        ],
        "max_loops": 3
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print("GroupChat swarm completed successfully!")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Chat discussion: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
  "status": "success",
  "swarm_name": "product-strategy-discussion", 
  "swarm_type": "GroupChat",
  "task": "Discuss and develop a go-to-market strategy for a new AI-powered productivity tool targeting small businesses",
  "output": {
    "chat_transcript": [
      {
        "agent": "Product Manager",
        "message": "Let's start by defining our target user. What specific pain points do small businesses have with productivity tools?"
      },
      {
        "agent": "UX Researcher", 
        "message": "From our research, small businesses struggle with tool complexity and time to value. They need something that works immediately without extensive setup."
      },
      {
        "agent": "Sales Director",
        "message": "I agree. SMBs have limited time and resources. They typically abandon tools that require more than a week to see value. Pricing is also critical - they're very cost-conscious."
      },
      {
        "agent": "Marketing Strategist",
        "message": "This suggests we should focus on 'instant productivity gains' messaging. We could position against complex enterprise tools that overwhelm small teams."
      }
    ],
    "key_decisions": [
      "Target: Small businesses with 5-50 employees",
      "Positioning: Simple, immediate productivity gains", 
      "Pricing: Freemium model with low-cost paid tiers",
      "GTM: Self-serve with strong onboarding"
    ],
    "final_strategy": "Launch with freemium model targeting productivity-focused small businesses through content marketing and self-serve channels..."
  },
  "metadata": {
    "conversation_rounds": 3,
    "total_messages": 12,
    "consensus_reached": true,
    "execution_time_seconds": 38.7,
    "billing_info": {
      "total_cost": 0.095
    }
  }
}
```

## Best Practices

- Set clear discussion goals and objectives
- Use diverse agent personalities for richer dialogue
- Allow multiple conversation rounds for idea development
- Encourage agents to build upon each other's contributions

## Related Swarm Types

- [MixtureOfAgents](mixture_of_agents.md) - For complementary expertise
- [MajorityVoting](majority_voting.md) - For consensus decision-making
- [AutoSwarmBuilder](auto_swarm_builder.md) - For automatic discussion setup