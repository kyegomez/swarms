# AutoSwarmBuilder

*Automatically configures optimal swarm architectures based on task requirements*

**Swarm Type**: `AutoSwarmBuilder`

## Overview

The AutoSwarmBuilder automatically configures optimal agent architectures based on task requirements and performance metrics, simplifying swarm creation. This intelligent system analyzes the given task and automatically generates the most suitable agent configuration, eliminating the need for manual swarm design.

Key features:
- **Intelligent Configuration**: Automatically designs optimal swarm structures
- **Task-Adaptive**: Adapts architecture based on specific task requirements
- **Performance Optimization**: Selects configurations for maximum efficiency
- **Simplified Setup**: Eliminates manual agent configuration complexity

## Use Cases

- Quick prototyping and experimentation
- Unknown or complex task requirements
- Automated swarm optimization
- Simplified swarm creation for non-experts

## API Usage

### Basic AutoSwarmBuilder Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Auto Marketing Campaign",
        "description": "Automatically build optimal swarm for marketing campaign creation",
        "swarm_type": "AutoSwarmBuilder",
        "task": "Create a comprehensive digital marketing campaign for a new sustainable fashion brand targeting Gen Z consumers",
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
        "name": "Auto Marketing Campaign",
        "description": "Automatically build optimal swarm for marketing campaign creation",
        "swarm_type": "AutoSwarmBuilder",
        "task": "Create a comprehensive digital marketing campaign for a new sustainable fashion brand targeting Gen Z consumers",
        "max_loops": 1
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print("AutoSwarmBuilder completed successfully!")
        print(f"Generated swarm architecture: {result['metadata']['generated_architecture']}")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Campaign output: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
  "status": "success",
  "swarm_name": "auto-marketing-campaign",
  "swarm_type": "AutoSwarmBuilder",
  "task": "Create a comprehensive digital marketing campaign for a new sustainable fashion brand targeting Gen Z consumers",
  "output": {
    "campaign_strategy": {
      "brand_positioning": "Authentic, sustainable fashion for conscious Gen Z consumers",
      "key_messaging": "Style that makes a difference - fashion with purpose",
      "target_demographics": "Ages 18-26, environmentally conscious, social media active"
    },
    "content_strategy": {
      "social_platforms": ["TikTok", "Instagram", "Pinterest"],
      "content_pillars": ["Sustainability education", "Style inspiration", "Behind-the-scenes"],
      "posting_schedule": "Daily posts across platforms with peak engagement timing"
    },
    "influencer_strategy": {
      "tier_1": "Micro-influencers (10K-100K followers) focused on sustainability",
      "tier_2": "Fashion nano-influencers (1K-10K followers) for authentic engagement",
      "collaboration_types": ["Product partnerships", "Brand ambassador programs"]
    },
    "paid_advertising": {
      "platforms": ["Instagram Ads", "TikTok Ads", "Google Ads"],
      "budget_allocation": "40% social media, 30% search, 30% video content",
      "targeting_strategy": "Interest-based and lookalike audiences"
    },
    "metrics_and_kpis": {
      "awareness": "Brand mention volume, reach, impressions",
      "engagement": "Comments, shares, saves, time spent",
      "conversion": "Website traffic, email signups, sales"
    }
  },
  "metadata": {
    "generated_architecture": {
      "selected_swarm_type": "MixtureOfAgents",
      "generated_agents": [
        "Brand Strategy Expert",
        "Gen Z Marketing Specialist", 
        "Social Media Content Creator",
        "Influencer Marketing Manager",
        "Digital Advertising Strategist"
      ],
      "reasoning": "Complex marketing campaign requires diverse expertise working collaboratively"
    },
    "auto_optimization": {
      "task_complexity": "High",
      "required_expertise_areas": 5,
      "optimal_architecture": "Collaborative with specialized agents"
    },
    "execution_time_seconds": 28.6,
    "billing_info": {
      "total_cost": 0.071
    }
  }
}
```

### Advanced Configuration

You can provide additional guidance to the AutoSwarmBuilder:

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Auto Research Project",
        "description": "Auto-build research swarm with specific constraints",
        "swarm_type": "AutoSwarmBuilder",
        "task": "Conduct comprehensive research on the impact of AI on healthcare outcomes",
        "rules": "Focus on peer-reviewed sources, include cost-benefit analysis, ensure balanced perspective on risks and benefits",
        "max_loops": 1
      }'
    ```

=== "Python (requests)"
    ```python
    swarm_config = {
        "name": "Auto Research Project",
        "description": "Auto-build research swarm with specific constraints",
        "swarm_type": "AutoSwarmBuilder", 
        "task": "Conduct comprehensive research on the impact of AI on healthcare outcomes",
        "rules": "Focus on peer-reviewed sources, include cost-benefit analysis, ensure balanced perspective on risks and benefits",
        "max_loops": 1
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    ```

## Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `task` | string | Task description for automatic optimization | Required |
| `rules` | string | Additional constraints and guidelines | None |
| `max_loops` | integer | Maximum execution rounds | 1 |

## Best Practices

- Provide detailed, specific task descriptions for better optimization
- Use `rules` parameter to guide the automatic configuration
- Ideal for rapid prototyping and experimentation
- Review generated architecture in response metadata

## Related Swarm Types

- [Auto](auto.md) - For automatic swarm type selection
- [MixtureOfAgents](mixture_of_agents.md) - Often selected by AutoSwarmBuilder
- [HierarchicalSwarm](hierarchical_swarm.md) - For complex structured tasks