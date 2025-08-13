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
          },
          {
            "agent_name": "Marketing Strategist", 
            "description": "Develops marketing and positioning strategy",
            "system_prompt": "You are a marketing strategist. Focus on target audience, messaging, channels, and competitive positioning. Contribute marketing insights to the discussion.",
            "model_name": "gpt-4o",
            "max_loops": 3,
          },
          {
            "agent_name": "Sales Director",
            "description": "Provides sales and customer perspective",
            "system_prompt": "You are a sales director with small business experience. Focus on pricing, sales process, customer objections, and market adoption. Share practical sales insights.",
            "model_name": "gpt-4o",
            "max_loops": 3,
          },
          {
            "agent_name": "UX Researcher",
            "description": "Represents user experience and research insights",
            "system_prompt": "You are a UX researcher specializing in small business tools. Focus on user behavior, usability, adoption barriers, and design considerations.",
            "model_name": "gpt-4o",
            "max_loops": 3,
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
            },
            {
                "agent_name": "Marketing Strategist", 
                "description": "Develops marketing and positioning strategy",
                "system_prompt": "You are a marketing strategist. Focus on target audience, messaging, channels, and competitive positioning. Contribute marketing insights to the discussion.",
                "model_name": "gpt-4o",
                "max_loops": 3,
            },
            {
                "agent_name": "Sales Director",
                "description": "Provides sales and customer perspective",
                "system_prompt": "You are a sales director with small business experience. Focus on pricing, sales process, customer objections, and market adoption. Share practical sales insights.",
                "model_name": "gpt-4o",
                "max_loops": 3,
            },
            {
                "agent_name": "UX Researcher",
                "description": "Represents user experience and research insights",
                "system_prompt": "You are a UX researcher specializing in small business tools. Focus on user behavior, usability, adoption barriers, and design considerations.",
                "model_name": "gpt-4o",
                "max_loops": 3,
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
    "job_id": "swarms-2COVtf3k0Fz7jU1BOOHF3b5nuL2x",
    "status": "success",
    "swarm_name": "Product Strategy Discussion",
    "description": "Collaborative chat to develop product strategy",
    "swarm_type": "GroupChat",
    "output": "User: \n\nSystem: \n Group Chat Name: Product Strategy Discussion\nGroup Chat Description: Collaborative chat to develop product strategy\n Agents in your Group Chat: Available Agents for Team: None\n\n\n\n[Agent 1]\nName: Product Manager\nDescription: Leads product strategy and development\nRole.....",
    "number_of_agents": 4,
    "service_tier": "standard",
    "execution_time": 47.36732482910156,
    "usage": {
        "input_tokens": 30,
        "output_tokens": 1633,
        "total_tokens": 1663,
        "billing_info": {
            "cost_breakdown": {
                "agent_cost": 0.04,
                "input_token_cost": 0.00009,
                "output_token_cost": 0.024495,
                "token_counts": {
                    "total_input_tokens": 30,
                    "total_output_tokens": 1633,
                    "total_tokens": 1663
                },
                "num_agents": 4,
                "service_tier": "standard",
                "night_time_discount_applied": false
            },
            "total_cost": 0.064585,
            "discount_active": false,
            "discount_type": "none",
            "discount_percentage": 0
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