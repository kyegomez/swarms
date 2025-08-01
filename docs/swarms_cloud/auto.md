# Auto

*Intelligently selects the most effective swarm architecture for a given task*

**Swarm Type**: `auto` (or `Auto`)

## Overview

The Auto swarm type intelligently selects the most effective swarm architecture for a given task based on context analysis and task requirements. This intelligent system evaluates the task description and automatically chooses the optimal swarm type from all available architectures, ensuring maximum efficiency and effectiveness.

Key features:
- **Intelligent Selection**: Automatically chooses the best swarm type for each task
- **Context Analysis**: Analyzes task requirements to make optimal decisions
- **Adaptive Architecture**: Adapts to different types of problems automatically
- **Zero Configuration**: No manual architecture selection required

## Use Cases

- When unsure about which swarm type to use
- General-purpose task automation
- Rapid prototyping and experimentation
- Simplified API usage for non-experts

## API Usage

### Basic Auto Example

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Auto Content Creation",
        "description": "Let the system choose the best approach for content creation",
        "swarm_type": "auto",
        "task": "Create a comprehensive blog post about sustainable investing, including research, writing, editing, and SEO optimization",
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
        "name": "Auto Content Creation",
        "description": "Let the system choose the best approach for content creation",
        "swarm_type": "auto",
        "task": "Create a comprehensive blog post about sustainable investing, including research, writing, editing, and SEO optimization",
        "max_loops": 1
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Auto swarm completed successfully!")
        print(f"Selected architecture: {result['metadata']['selected_swarm_type']}")
        print(f"Cost: ${result['metadata']['billing_info']['total_cost']}")
        print(f"Execution time: {result['metadata']['execution_time_seconds']} seconds")
        print(f"Content: {result['output']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    ```

**Example Response**:
```json
{
  "status": "success",
  "swarm_name": "auto-content-creation",
  "swarm_type": "SequentialWorkflow",
  "task": "Create a comprehensive blog post about sustainable investing, including research, writing, editing, and SEO optimization",
  "output": {
    "research_phase": {
      "key_findings": "Sustainable investing has grown 42% in the past two years...",
      "market_trends": "ESG funds outperformed traditional funds by 3.2%...",
      "statistics": "Global sustainable investment assets reached $35.3 trillion..."
    },
    "writing_phase": {
      "title": "The Future of Sustainable Investing: A Guide to ESG Strategies",
      "content": "Comprehensive blog post with introduction, main sections, and conclusion...",
      "word_count": 1850
    },
    "editing_phase": {
      "improvements": "Enhanced clarity, improved flow, corrected grammar",
      "readability_score": "Grade 8 level - accessible to general audience",
      "final_content": "Polished blog post ready for publication..."
    },
    "seo_optimization": {
      "target_keywords": ["sustainable investing", "ESG funds", "green finance"],
      "meta_description": "Discover the future of sustainable investing...",
      "optimized_content": "SEO-optimized version with strategic keyword placement"
    }
  },
  "metadata": {
    "auto_selection": {
      "selected_swarm_type": "SequentialWorkflow",
      "reasoning": "Task requires step-by-step content creation process where each phase builds on the previous",
      "analysis": {
        "task_complexity": "Medium-High",
        "sequential_dependencies": true,
        "parallel_opportunities": false,
        "collaboration_needs": "Low"
      }
    },
    "generated_agents": [
      "Research Specialist",
      "Content Writer", 
      "Editor",
      "SEO Optimizer"
    ],
    "execution_time_seconds": 43.2,
    "billing_info": {
      "total_cost": 0.087
    }
  }
}
```

### Advanced Auto Usage

You can provide additional context to help the Auto selection:

=== "Shell (curl)"
    ```bash
    curl -X POST "https://api.swarms.world/v1/swarm/completions" \
      -H "x-api-key: $SWARMS_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "name": "Auto Business Analysis",
        "description": "Automatic swarm selection for business analysis",
        "swarm_type": "auto",
        "task": "Analyze market opportunities for a new AI startup in healthcare",
        "rules": "Need multiple perspectives from different business functions, time-sensitive analysis required",
        "max_loops": 1
      }'
    ```

=== "Python (requests)"
    ```python
    swarm_config = {
        "name": "Auto Business Analysis",
        "description": "Automatic swarm selection for business analysis",
        "swarm_type": "auto",
        "task": "Analyze market opportunities for a new AI startup in healthcare",
        "rules": "Need multiple perspectives from different business functions, time-sensitive analysis required",
        "max_loops": 1
    }
    
    response = requests.post(
        f"{API_BASE_URL}/v1/swarm/completions",
        headers=headers,
        json=swarm_config
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Auto selected: {result['metadata']['auto_selection']['selected_swarm_type']}")
        print(f"Reasoning: {result['metadata']['auto_selection']['reasoning']}")
    ```

## Selection Logic

The Auto swarm type analyzes various factors to make its selection:

| Factor | Consideration |
|--------|---------------|
| **Task Complexity** | Simple → Single agent, Complex → Multi-agent |
| **Sequential Dependencies** | Dependencies → SequentialWorkflow |
| **Parallel Opportunities** | Independent subtasks → ConcurrentWorkflow |
| **Collaboration Needs** | Discussion required → GroupChat |
| **Expertise Diversity** | Multiple domains → MixtureOfAgents |
| **Management Needs** | Oversight required → HierarchicalSwarm |
| **Routing Requirements** | Task distribution → MultiAgentRouter |

## Best Practices

- Provide detailed task descriptions for better selection
- Use `rules` parameter to guide selection criteria
- Review the selected architecture in response metadata
- Ideal for users new to swarm architectures

## Related Swarm Types

Since Auto can select any swarm type, it's related to all architectures:
- [AutoSwarmBuilder](auto_swarm_builder.md) - For automatic agent generation
- [SequentialWorkflow](sequential_workflow.md) - Often selected for linear tasks
- [ConcurrentWorkflow](concurrent_workflow.md) - For parallel processing needs
- [MixtureOfAgents](mixture_of_agents.md) - For diverse expertise requirements