# SwarmRouter Examples

The SwarmRouter is a flexible routing system designed to manage different types of swarms for task execution. It provides a unified interface to interact with various swarm types, including `AgentRearrange`, `MixtureOfAgents`, `SpreadSheetSwarm`, `SequentialWorkflow`, and `ConcurrentWorkflow`.

## Prerequisites

- Python 3.7+
- OpenAI API key or other supported LLM provider keys
- Swarms library

## Installation

```bash
pip3 install -U swarms
```

## Environment Variables

```plaintext
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GROQ_API_KEY=""
```

## Basic Usage

### 1. Initialize Specialized Agents

```python
from swarms import Agent
from swarms.structs.swarm_router import SwarmRouter

# Initialize specialized agents
data_extractor_agent = Agent(
    agent_name="Data-Extractor",
    system_prompt="You are a data extraction specialist...",
    model_name="gpt-4.1",
    max_loops=1,
)

summarizer_agent = Agent(
    agent_name="Document-Summarizer",
    system_prompt="You are a document summarization expert...",
    model_name="gpt-4.1",
    max_loops=1,
)

financial_analyst_agent = Agent(
    agent_name="Financial-Analyst",
    system_prompt="You are a financial analysis specialist...",
    model_name="gpt-4.1",
    max_loops=1,
)
```

### 2. Create SwarmRouter with Sequential Workflow

```python
sequential_router = SwarmRouter(
    name="SequentialRouter",
    description="Process tasks in sequence",
    agents=[data_extractor_agent, summarizer_agent, financial_analyst_agent],
    swarm_type="SequentialWorkflow",
    max_loops=1
)

# Run a task
result = sequential_router.run("Analyze and summarize the quarterly financial report")
```

### 3. Create SwarmRouter with Concurrent Workflow

```python
concurrent_router = SwarmRouter(
    name="ConcurrentRouter",
    description="Process tasks concurrently",
    agents=[data_extractor_agent, summarizer_agent, financial_analyst_agent],
    swarm_type="ConcurrentWorkflow",
    max_loops=1
)

# Run a task
result = concurrent_router.run("Evaluate multiple aspects of the company simultaneously")
```

### 4. Create SwarmRouter with AgentRearrange

```python
rearrange_router = SwarmRouter(
    name="RearrangeRouter",
    description="Dynamically rearrange agents for optimal task processing",
    agents=[data_extractor_agent, summarizer_agent, financial_analyst_agent],
    swarm_type="AgentRearrange",
    rearrange_flow=f"{data_extractor_agent.agent_name} -> {summarizer_agent.agent_name} -> {financial_analyst_agent.agent_name}",
    max_loops=1
)

# Run a task
result = rearrange_router.run("Process and analyze company documents")
```

### 5. Create SwarmRouter with MixtureOfAgents

```python
mixture_router = SwarmRouter(
    name="MixtureRouter",
    description="Combine multiple expert agents",
    agents=[data_extractor_agent, summarizer_agent, financial_analyst_agent],
    swarm_type="MixtureOfAgents",
    max_loops=1
)

# Run a task
result = mixture_router.run("Provide comprehensive analysis of company performance")
```

## Advanced Features

### 1. Error Handling and Logging

```python
try:
    result = router.run("Complex analysis task")
    
    # Retrieve and print logs
    for log in router.get_logs():
        print(f"{log.timestamp} - {log.level}: {log.message}")
except Exception as e:
    print(f"Error occurred: {str(e)}")
```

### 2. Custom Configuration with Autosave

```python
router = SwarmRouter(
    name="CustomRouter",
    description="Custom router configuration",
    agents=[data_extractor_agent, summarizer_agent, financial_analyst_agent],
    swarm_type="SequentialWorkflow",
    max_loops=3,
    autosave=True,  # Enable automatic saving of config, state, and metadata
    verbose=True,
    output_type="json"
)

# When autosave is enabled:
# - config.json is saved on initialization to workspace_dir/swarms/SwarmRouter/CustomRouter-{timestamp}/
# - state.json and metadata.json are saved after each run() call
result = router.run("Analyze the investment opportunity")
```

# SwarmType Reference

## Valid SwarmType Values

| Value | Description |
|-------|-------------|
| `"SequentialWorkflow"` | Execute agents in sequence |
| `"ConcurrentWorkflow"` | Execute agents concurrently |
| `"AgentRearrange"` | Dynamically rearrange agent execution order |
| `"MixtureOfAgents"` | Combine outputs from multiple agents |
| `"GroupChat"` | Enable group chat between agents |
| `"MultiAgentRouter"` | Route tasks to appropriate agents |
| `"AutoSwarmBuilder"` | Automatically build swarm configuration |
| `"HiearchicalSwarm"` | Hierarchical agent organization |
| `"MajorityVoting"` | Use majority voting for decisions |
| `"CouncilAsAJudge"` | Council-based evaluation system |
| `"HeavySwarm"` | Heavy swarm for complex tasks |
| `"BatchedGridWorkflow"` | Batched grid workflow for parallel task processing |
| `"LLMCouncil"` | Council of specialized LLM agents with peer review |
| `"DebateWithJudge"` | Debate architecture with Pro/Con agents and a Judge |
| `"RoundRobin"` | Round-robin execution cycling through agents |
| `"auto"` | Automatically select swarm type |

# Best Practices

## Choose the appropriate swarm type based on your task requirements:

| Swarm Type | Use Case |
|------------|----------|
| `SequentialWorkflow` | Tasks that need to be processed in order |
| `ConcurrentWorkflow` | Independent tasks that can be processed simultaneously |
| `AgentRearrange` | Tasks requiring dynamic agent organization |
| `MixtureOfAgents` | Complex tasks needing multiple expert perspectives |

## Configure agents appropriately:

   | Configuration Aspect | Description |
   |---------------------|-------------|
   | Agent Names & Descriptions | Set meaningful and descriptive names that reflect the agent's role and purpose |
   | System Prompts | Define clear, specific prompts that outline the agent's responsibilities and constraints |
   | Model Parameters | Configure appropriate parameters like temperature, max_tokens, and other model-specific settings |

## Implement proper error handling:

| Error Handling Practice | Description |
|------------------------|-------------|
| Try-Except Blocks | Implement proper exception handling with try-except blocks |
| Log Monitoring | Regularly monitor and analyze system logs for potential issues |
| Edge Case Handling | Implement specific handling for edge cases and unexpected scenarios |

## Optimize performance:

| Performance Optimization | Description |
|------------------------|-------------|
| Concurrent Processing | Utilize parallel processing capabilities when tasks can be executed simultaneously |
| Max Loops Configuration | Set appropriate iteration limits based on task complexity and requirements |
| Resource Management | Continuously monitor and optimize system resource utilization |

## Example Implementation

Here's a complete example showing how to use SwarmRouter in a real-world scenario:

```python
import os
from swarms import Agent
from swarms.structs.swarm_router import SwarmRouter

# Initialize specialized agents
research_agent = Agent(
    agent_name="ResearchAgent",
    system_prompt="You are a research specialist...",
    model_name="gpt-4.1",
    max_loops=1
)

analysis_agent = Agent(
    agent_name="AnalysisAgent",
    system_prompt="You are an analysis expert...",
    model_name="gpt-4.1",
    max_loops=1
)

summary_agent = Agent(
    agent_name="SummaryAgent",
    system_prompt="You are a summarization specialist...",
    model_name="gpt-4.1",
    max_loops=1
)

# Create router with sequential workflow
router = SwarmRouter(
    name="ResearchAnalysisRouter",
    description="Process research and analysis tasks",
    agents=[research_agent, analysis_agent, summary_agent],
    swarm_type="SequentialWorkflow",
    max_loops=1,
    verbose=True
)

# Run complex task
try:
    result = router.run(
        "Research and analyze the impact of AI on healthcare, "
        "providing a comprehensive summary of findings."
    )
    print("Task Result:", result)
    
    # Print logs
    for log in router.get_logs():
        print(f"{log.timestamp} - {log.level}: {log.message}")
        
except Exception as e:
    print(f"Error processing task: {str(e)}")
```

This comprehensive guide demonstrates how to effectively use the SwarmRouter in various scenarios, making it easier to manage and orchestrate multiple agents for complex tasks. 